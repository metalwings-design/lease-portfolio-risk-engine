"""gold.py

Gold layer assets for the Athena equipment finance platform.
Implements executive portfolio analytics and regulatory risk metrics:
1. gold_portfolio_concentration_hhi: Exposure concentration analysis (HHI).
2. gold_vintage_delinquency_matrix: Static pool cumulative vintage delinquency curves.
3. gold_macro_stress_sensitivity: Dynamic rate shock stress testing and effective duration.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dagster import Output, asset


def get_project_root() -> Path:
    """Resolve project root directory containing data and reports folders."""
    curr = Path(__file__).resolve()
    for parent in curr.parents:
        if (parent / "data").is_dir():
            return parent
    return Path.cwd().resolve()


FIGURES_DIR = get_project_root() / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@asset(
    group_name="gold",
    description="Portfolio exposure concentration analysis and Herfindahl-Hirschman Index (HHI)."
)
def gold_portfolio_concentration_hhi(
    silver_contracts_augmented: pd.DataFrame,
    silver_cashflows_augmented: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Computes exposure distribution, HHI concentration scores, and regulatory

    classifications across Equipment Category, Risk Tier, and Geographic State.
    """
    df_c = silver_contracts_augmented.copy()
    total_exposure = df_c["net_investment_origination"].sum()

    dimensions = [
        ("Equipment Category", "equipment_category"),
        ("Risk Tier", "risk_tier"),
        ("Geographic State", "state"),
    ]

    records = []
    hhi_scores = {}

    for dim_type, col_name in dimensions:
        grp = df_c.groupby(col_name)["net_investment_origination"].sum().reset_index()
        grp["exposure_inr_cr"] = grp["net_investment_origination"] / 1e7
        grp["exposure_weight_pct"] = (grp["net_investment_origination"] / total_exposure) * 100.0
        hhi = float((grp["exposure_weight_pct"] ** 2).sum())
        hhi_scores[dim_type] = hhi

        # Regulatory classification
        if hhi < 1500.0:
            classification = "Low Concentration / Diversified"
        elif hhi <= 2500.0:
            classification = "Moderate Concentration"
        else:
            classification = "High Concentration"

        for _, r in grp.iterrows():
            records.append({
                "dimension_type": dim_type,
                "dimension_value": str(r[col_name]),
                "exposure_inr_cr": float(r["exposure_inr_cr"]),
                "exposure_weight_pct": float(r["exposure_weight_pct"]),
                "dimension_hhi": round(hhi, 2),
                "concentration_classification": classification,
            })

    result_df = pd.DataFrame(records)

    # ----------------------------------------------------
    # Visualization: Horizontal Bar Charts
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cat_df = result_df[result_df["dimension_type"] == "Equipment Category"].sort_values("exposure_weight_pct")
    ax1.barh(cat_df["dimension_value"], cat_df["exposure_weight_pct"], color="#1f4e79", edgecolor="none", height=0.6)
    ax1.set_title(f"Equipment Category (HHI: {hhi_scores['Equipment Category']:.1f})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Portfolio Weight (%)", fontsize=10)
    for i, v in enumerate(cat_df["exposure_weight_pct"]):
        ax1.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    state_df = result_df[result_df["dimension_type"] == "Geographic State"].sort_values("exposure_weight_pct")
    ax2.barh(state_df["dimension_value"], state_df["exposure_weight_pct"], color="#2e7d32", edgecolor="none", height=0.6)
    ax2.set_title(f"Geographic State (HHI: {hhi_scores['Geographic State']:.1f})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Portfolio Weight (%)", fontsize=10)
    for i, v in enumerate(state_df["exposure_weight_pct"]):
        ax2.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.suptitle("Athena Portfolio Exposure Concentration & HHI Metrics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = FIGURES_DIR / "portfolio_concentration_hhi.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return Output(
        result_df,
        metadata={
            "hhi_equipment_category": float(hhi_scores.get("Equipment Category", 0.0)),
            "hhi_risk_tier": float(hhi_scores.get("Risk Tier", 0.0)),
            "hhi_geographic_state": float(hhi_scores.get("Geographic State", 0.0)),
            "chart_saved_path": str(plot_path),
        },
    )


@asset(
    group_name="gold",
    description="Vintage static pool delinquency matrix and cumulative default curves across Months on Book (MOB)."
)
def gold_vintage_delinquency_matrix(
    silver_contracts_augmented: pd.DataFrame,
    silver_cashflows_augmented: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Constructs static pool vintage delinquency schedules tracking cumulative

    default rates over contract lifecycle (MOB 1 to 60).
    """
    df_c = silver_contracts_augmented[["contract_id", "origination_vintage", "financed_amount"]].copy()
    df_cf = silver_cashflows_augmented[["contract_id", "installment_num", "dpd_status", "is_defaulted"]].copy()

    # Identify default event (90+ DPD or is_defaulted == 1)
    df_cf["is_def"] = ((df_cf["dpd_status"] == "90_PLUS_DEFAULT") | (df_cf["is_defaulted"] == 1)).astype(int)

    # Merge vintage and financed principal
    merged = df_cf.merge(df_c, on="contract_id", how="left")

    cohort_totals = df_c.groupby("origination_vintage")["financed_amount"].sum() / 1e7

    # Find earliest installment where default occurred for each contract
    def_events = merged[merged["is_def"] == 1].groupby("contract_id")["installment_num"].min()
    contract_financed_dict = df_c.set_index("contract_id")["financed_amount"].to_dict()

    vintages = sorted(merged["origination_vintage"].unique())
    records = []

    for v in vintages:
        v_cf = merged[merged["origination_vintage"] == v]
        cohort_total_cr = float(cohort_totals[v])
        max_mob = int(v_cf["installment_num"].max())

        # Contracts belonging to this vintage
        v_contracts = set(v_cf["contract_id"].unique())
        v_def_contracts = def_events[def_events.index.isin(v_contracts)]

        cum_default_cr = 0.0
        for mob in range(1, max_mob + 1):
            if not v_def_contracts.empty:
                hit_contracts = v_def_contracts[v_def_contracts <= mob].index
                defaulted_amt = sum(contract_financed_dict[cid] for cid in hit_contracts) / 1e7
                cum_default_cr = max(cum_default_cr, defaulted_amt)
            else:
                cum_default_cr = 0.0

            rate_pct = (cum_default_cr / cohort_total_cr * 100.0) if cohort_total_cr > 0 else 0.0
            records.append({
                "origination_vintage": v,
                "months_on_book": mob,
                "cohort_total_financed_cr": round(cohort_total_cr, 4),
                "cumulative_defaulted_cr": round(cum_default_cr, 4),
                "cumulative_default_rate_pct": round(rate_pct, 4),
            })

    result_df = pd.DataFrame(records)

    # Enforce strict monotonicity per vintage cohort
    result_df["cumulative_default_rate_pct"] = (
        result_df.groupby("origination_vintage")["cumulative_default_rate_pct"].cummax()
    )

    # ----------------------------------------------------
    # Visualization: Vintage Curves across MOB
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Highlight recent annual cohorts (e.g. 2021 through 2024 cohorts)
    plot_vintages = [v for v in vintages if v.startswith(("2021", "2022", "2023", "2024"))]
    cmap = plt.colormaps["viridis"].resampled(max(len(plot_vintages), 1))

    for idx, v in enumerate(plot_vintages):
        sub = result_df[result_df["origination_vintage"] == v]
        ax.plot(
            sub["months_on_book"],
            sub["cumulative_default_rate_pct"],
            label=v,
            color=cmap(idx),
            linewidth=1.6,
            alpha=0.85,
        )

    ax.set_title("Multi-Vintage Cumulative Delinquency Curves (MOB 1 to 60)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Months on Book (MOB)", fontsize=10)
    ax.set_ylabel("Cumulative Default Rate (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Vintage", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)

    plt.tight_layout()
    plot_path = FIGURES_DIR / "vintage_curves.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    max_delinquency = float(result_df["cumulative_default_rate_pct"].max())
    latest_vintage = str(vintages[-1])
    latest_perf = float(
        result_df[result_df["origination_vintage"] == latest_vintage]["cumulative_default_rate_pct"].max()
    )

    return Output(
        result_df,
        metadata={
            "total_matrix_rows": int(len(result_df)),
            "vintage_cohorts_count": int(len(vintages)),
            "portfolio_max_delinquency_pct": max_delinquency,
            "latest_vintage_cohort": latest_vintage,
            "latest_vintage_max_default_pct": latest_perf,
            "chart_saved_path": str(plot_path),
        },
    )


@asset(
    group_name="gold",
    description="Macro interest rate stress testing, mark-to-market P&L sensitivity, and effective duration."
)
def gold_macro_stress_sensitivity(
    silver_cashflows_augmented: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Simulates 5 parallel yield curve shocks (+/-200 bps) to evaluate portfolio

    Fair Value impact, MTM gain/loss, and portfolio effective duration.
    """
    df_cf = silver_cashflows_augmented.copy()

    # Extract active/future cash flows (from current portfolio cutoff date 2024-12 onward)
    active_cf = df_cf[df_cf["reporting_month"] >= "2024-12"].copy()

    # Calculate remaining installment horizon k - t + 1 (1, 2, ..., n) for active contracts
    active_cf["horizon_m"] = active_cf.groupby("contract_id").cumcount() + 1

    cfs = active_cf["expected_cashflow"].values
    horizon = active_cf["horizon_m"].values
    r_base_annual = active_cf["r_market_annual"].values

    scenarios = [
        ("Severe Cut (-200 bps)", -200, -0.02),
        ("Mild Cut (-100 bps)", -100, -0.01),
        ("Base Scenario", 0, 0.00),
        ("Mild Hike (+100 bps)", 100, 0.01),
        ("Severe Hike (+200 bps)", 200, 0.02),
    ]

    fv_results = {}
    for name, bps, delta_r in scenarios:
        r_scen_monthly = (r_base_annual + delta_r) / 12.0
        # Discount factor df = (1 + r)^(-horizon)
        discount_factors = (1.0 + r_scen_monthly) ** (-horizon)
        fv_total = float(np.sum(cfs * discount_factors))
        fv_results[bps] = fv_total

    base_fv = fv_results[0]
    records = []

    for name, bps, delta_r in scenarios:
        fv = fv_results[bps]
        pnl = fv - base_fv
        pct_change = (pnl / base_fv) * 100.0 if base_fv > 0 else 0.0
        records.append({
            "scenario_name": name,
            "rate_shock_bps": bps,
            "revalued_portfolio_fv_cr": round(fv / 1e7, 4),
            "pnl_impact_cr": round(pnl / 1e7, 4),
            "pct_value_change": round(pct_change, 4),
        })

    result_df = pd.DataFrame(records)

    # Compute Portfolio Effective Duration: D_eff = (FV_-100 - FV_+100) / (2 * FV_0 * 0.01)
    d_eff = (fv_results[-100] - fv_results[100]) / (2.0 * base_fv * 0.01)

    # ----------------------------------------------------
    # Visualization: Scenario P&L Impact Waterfall / Bar
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    plot_order = [
        "Severe Cut (-200 bps)",
        "Mild Cut (-100 bps)",
        "Base Scenario",
        "Mild Hike (+100 bps)",
        "Severe Hike (+200 bps)",
    ]
    plot_df = result_df.set_index("scenario_name").loc[plot_order].reset_index()

    bar_colors = [
        "#2e7d32" if x > 0 else ("#616161" if x == 0 else "#c62828")
        for x in plot_df["pnl_impact_cr"]
    ]

    bars = ax.bar(plot_df["scenario_name"], plot_df["pnl_impact_cr"], color=bar_colors, width=0.52)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_title("Portfolio MTM P&L Impact Under Macro Interest Rate Shocks", fontsize=12, fontweight="bold")
    ax.set_ylabel("MTM Valuation Gain / Loss (₹ Crore)", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Value callouts on bars
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.25 if h >= 0 else -0.55
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f"₹{h:+.2f} Cr\n({h / (base_fv / 1e7) * 100:+.1f}%)",
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="bold",
        )

    plt.xticks(rotation=12, fontsize=9)
    plt.tight_layout()
    plot_path = FIGURES_DIR / "macro_rate_shock_sensitivity.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return Output(
        result_df,
        metadata={
            "base_portfolio_fv_cr": round(base_fv / 1e7, 4),
            "effective_duration_years": round(float(d_eff), 2),
            "severe_hike_pnl_cr": float(result_df.loc[result_df["rate_shock_bps"] == 200, "pnl_impact_cr"].iloc[0]),
            "severe_cut_pnl_cr": float(result_df.loc[result_df["rate_shock_bps"] == -200, "pnl_impact_cr"].iloc[0]),
            "chart_saved_path": str(plot_path),
        },
    )
