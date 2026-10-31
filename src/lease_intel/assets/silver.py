"""silver.py

Silver layer assets for the Athena equipment finance pipeline.
Implements derived tables:
1. silver_rbi_macro_monthly: Unified monthly macro benchmark table.
2. silver_contracts_augmented: Enriched contract master table with macro spreads,
   unearned finance ratios, and baseline expected loss at origination.
3. silver_cashflows_augmented: Granular monthly cash-flow valuation schedule
   computing contractual PV (ASC 842 / IFRS 16), dynamic fair value MTM,
   and interest margin compression.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from dagster import Output, asset


@asset(
    group_name="silver",
    description="Unified macroeconomic benchmark table keyed by reporting_month (YYYY-MM)."
)
def silver_rbi_macro_monthly(
    bronze_rbi_policy_rates: pd.DataFrame,
    bronze_rbi_call_money: pd.DataFrame,
    bronze_rbi_tbill_yields: pd.DataFrame,
    bronze_rbi_nbfc_credit: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Combines monetary policy rates, call money rates, T-Bill yield curves,

    and sectoral NBFC credit growth into a single monthly panel.
    """
    # Extract NBFC sector benchmark growth and outstanding balances
    nbfc_slice = bronze_rbi_nbfc_credit[
        bronze_rbi_nbfc_credit["sector_name"] == "Non-Banking Financial Companies (NBFCs)"
    ][["reporting_month", "outstanding_inr_crore", "yoy_growth_pct"]].rename(
        columns={
            "outstanding_inr_crore": "nbfc_credit_outstanding_crore",
            "yoy_growth_pct": "nbfc_credit_yoy_pct",
        }
    )

    # Outer merge across all four macroeconomic tables
    df_macro = bronze_rbi_policy_rates.merge(bronze_rbi_call_money, on="reporting_month", how="outer")
    df_macro = df_macro.merge(bronze_rbi_tbill_yields, on="reporting_month", how="outer")
    df_macro = df_macro.merge(nbfc_slice, on="reporting_month", how="outer")

    # Sort chronologically
    df_macro = df_macro.sort_values("reporting_month").reset_index(drop=True)

    # Handle reporting gaps by forward-filling and back-filling rate metrics
    rate_cols = [
        "repo_rate", "reverse_repo_rate", "bank_rate", "call_money_rate",
        "tbill_yield_14d", "tbill_yield_91d", "tbill_yield_182d", "tbill_yield_364d",
    ]
    df_macro[rate_cols] = df_macro[rate_cols].ffill().bfill()

    # Ensure explicit numeric float dtypes
    for col in df_macro.columns:
        if col != "reporting_month":
            df_macro[col] = pd.to_numeric(df_macro[col], errors="coerce")

    # Compute yield curve slope: 364-day T-Bill secondary yield vs. RBI Policy Repo Rate
    df_macro["yield_curve_slope_364d_vs_repo"] = df_macro["tbill_yield_364d"] - df_macro["repo_rate"]

    return Output(
        df_macro,
        metadata={
            "total_months": int(len(df_macro)),
            "earliest_month": str(df_macro["reporting_month"].min()),
            "latest_month": str(df_macro["reporting_month"].max()),
        },
    )


@asset(
    group_name="silver",
    description="Contract master table enriched with macro inception spreads, credit metrics, and initial valuation."
)
def silver_contracts_augmented(
    bronze_contracts: pd.DataFrame,
    silver_rbi_macro_monthly: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Augments synthetic lease contracts with macro conditions at origination,

    pricing spreads over benchmarks, and baseline expected loss.
    """
    # Join contracts to macro on origination month
    df = bronze_contracts.merge(
        silver_rbi_macro_monthly,
        left_on="origination_month",
        right_on="reporting_month",
        how="left",
        suffixes=("", "_macro"),
    )

    # 1. Inception Pricing Spreads
    df["inception_spread_over_repo"] = (df["contractual_rate"] * 100.0) - df["repo_rate"]
    df["inception_spread_over_364d"] = (df["contractual_rate"] * 100.0) - df["tbill_yield_364d"]

    # 2. Net Investment Balance at Origination
    df["net_investment_origination"] = df["financed_amount"]
    df["unearned_finance_ratio"] = df["unearned_finance_income"] / df["gross_investment"]

    # 3. Baseline Expected Loss at Inception (LGD = 30%, Collateral Recovery = 70%)
    df["expected_loss_origination"] = df["financed_amount"] * df["baseline_annual_pd"] * 0.30

    return Output(
        df,
        metadata={
            "contract_count": int(len(df)),
            "mean_inception_spread": float(df["inception_spread_over_repo"].mean()),
            "total_financed_amount": float(df["financed_amount"].sum()),
        },
    )


@asset(
    group_name="silver",
    description="Granular monthly cash-flow valuation schedule computing contractual PV, market Fair Value, and MTM variance."
)
def silver_cashflows_augmented(
    bronze_monthly_cashflows: pd.DataFrame,
    silver_contracts_augmented: pd.DataFrame,
    silver_rbi_macro_monthly: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Enriches cash flows with contractual present value (ASC 842 / IFRS 16),

    dynamic risk-adjusted mark-to-market fair value, and interest margin compression.
    """
    # Inherit contract parameters
    contract_cols = ["contract_id", "contractual_rate", "term_months", "equipment_category", "risk_tier"]
    df = bronze_monthly_cashflows.merge(silver_contracts_augmented[contract_cols], on="contract_id", how="left")

    # Inherit prevailing macro rates for the cash flow reporting month
    macro_cols = ["reporting_month", "repo_rate", "tbill_yield_364d"]
    df = df.merge(silver_rbi_macro_monthly[macro_cols], on="reporting_month", how="left")

    # For cash flows past latest reporting month, forward fill prevailing rates
    df["repo_rate"] = df["repo_rate"].ffill()
    df["tbill_yield_364d"] = df["tbill_yield_364d"].ffill()

    # Dynamic Fair Value Benchmark Discount Rate: r_market = T-Bill 364d + Spread(Risk Tier)
    tier_spreads = {"Tier 1": 0.020, "Tier 2": 0.035, "Tier 3": 0.055}
    credit_spread = df["risk_tier"].map(tier_spreads).fillna(0.035)
    df["r_market_annual"] = (df["tbill_yield_364d"] / 100.0) + credit_spread

    # Ensure sorted by contract_id and installment_num
    df = df.sort_values(["contract_id", "installment_num"]).reset_index(drop=True)

    # Compute Contractual PV and Market FV for remaining installments k in [t, n]
    pv_contractual = np.empty(len(df), dtype=np.float64)
    fv_market = np.empty(len(df), dtype=np.float64)

    contracts_u, idx_start, counts = np.unique(df["contract_id"].values, return_index=True, return_counts=True)
    cfs_all = df["expected_cashflow"].values
    r_c_all = df["contractual_rate"].values / 12.0
    r_m_all = df["r_market_annual"].values / 12.0

    for start, n in zip(idx_start, counts):
        end = start + n
        cfs = cfs_all[start:end]
        r_c = r_c_all[start]
        r_m = r_m_all[start:end]

        # Contractual PV: df_c = (1 + r_c)^(-(k - t + 1))
        t_idx = np.arange(1, n + 1)
        w_c = cfs * ((1.0 + r_c) ** (-t_idx))
        pv_contractual[start:end] = np.cumsum(w_c[::-1])[::-1] * ((1.0 + r_c) ** (t_idx - 1))

        # Dynamic Market FV: df_m = (1 + r_m[t])^(-(k - t + 1))
        for i in range(n):
            rem_len = n - i
            k_range = np.arange(1, rem_len + 1)
            fv_market[start + i] = np.sum(cfs[i:] / ((1.0 + r_m[i]) ** k_range))

    df["pv_contractual"] = pv_contractual
    df["fv_market"] = fv_market
    df["valuation_variance"] = fv_market - pv_contractual
    df["current_interest_margin"] = (df["contractual_rate"] * 100.0) - df["repo_rate"]

    return Output(
        df,
        metadata={
            "cashflow_records_count": int(len(df)),
            "total_pv_contractual": float(df["pv_contractual"].sum()),
            "total_fv_market": float(df["fv_market"].sum()),
        },
    )
