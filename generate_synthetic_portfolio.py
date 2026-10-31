from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Directory setup
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "synthetic_leases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Calibration Specs based on Indian NBFC Market & RBI Sectors
# ---------------------------------------------------------
CATEGORY_SPECS = {
    "Commercial Vehicles": {
        "cost_min": 1_500_000,
        "cost_max": 6_500_000,
        "tenors": [36, 48],
        "urv_pct_range": (0.10, 0.15),
        "rate_range": (0.105, 0.135),
        "weight": 0.45,
    },
    "Construction Equipment": {
        "cost_min": 4_500_000,
        "cost_max": 25_000_000,
        "tenors": [36, 48, 60],
        "urv_pct_range": (0.15, 0.25),
        "rate_range": (0.110, 0.145),
        "weight": 0.30,
    },
    "Agriculture Equipment": {
        "cost_min": 600_000,
        "cost_max": 3_000_000,
        "tenors": [24, 36, 48],
        "urv_pct_range": (0.10, 0.20),
        "rate_range": (0.095, 0.125),
        "weight": 0.25,
    },
}

INDIAN_STATES = [
    "Maharashtra", "Gujarat", "Tamil Nadu", "Karnataka", 
    "Uttar Pradesh", "Rajasthan", "Madhya Pradesh", "Andhra Pradesh"
]
STATE_WEIGHTS = [0.22, 0.18, 0.14, 0.12, 0.11, 0.09, 0.08, 0.06]

def calc_lease_pmt(pv: float, fv: float, r_m: float, n: int) -> float:
    """Calculates contractual monthly annuity payment with a residual balloon value."""
    if r_m == 0:
        return (pv - fv) / n
    discount_fv = fv / ((1.0 + r_m) ** n)
    pmt = (pv - discount_fv) * (r_m / (1.0 - (1.0 + r_m) ** (-n)))
    return pmt

def generate_portfolio(n_contracts: int = 1200, seed: int = 42):
    np.random.seed(seed)
    categories = list(CATEGORY_SPECS.keys())
    cat_weights = [CATEGORY_SPECS[c]["weight"] for c in categories]

    # Origination cohorts spanning 48 continuous months (Jan 2021 to Dec 2024)
    start_date = pd.to_datetime("2010-01-01")
    end_date = pd.to_datetime("2024-12-31")
    days_range = (end_date - start_date).days

    contracts = []
    cashflow_rows = []

    print(f"Generating {n_contracts} lease contracts and cash flow amortization schedules...")

    for i in range(1, n_contracts + 1):
        contract_id = f"EQ-{i:05d}"
        cat = np.random.choice(categories, p=cat_weights)
        spec = CATEGORY_SPECS[cat]

        # Asset & Loan terms
        asset_cost = round(float(np.random.uniform(spec["cost_min"], spec["cost_max"])), -3)
        down_pmt_pct = np.random.uniform(0.15, 0.25)
        financed_amount = round(asset_cost * (1.0 - down_pmt_pct), 2)
        
        term_months = int(np.random.choice(spec["tenors"]))
        urv_pct = np.random.uniform(*spec["urv_pct_range"])
        residual_value = round(asset_cost * urv_pct, 2)
        
        contractual_rate = round(float(np.random.uniform(*spec["rate_range"])), 4)
        r_monthly = contractual_rate / 12.0
        
        monthly_payment = round(calc_lease_pmt(financed_amount, residual_value, r_monthly, term_months), 2)
        gross_investment = round((monthly_payment * term_months) + residual_value, 2)
        unearned_finance_income = round(gross_investment - financed_amount, 2)

        # Commencement Date and Cohort
        random_days = int(np.random.uniform(0, days_range))
        commence = start_date + timedelta(days=random_days)
        commence_date = commence.date()
        orig_month = commence.strftime("%Y-%m")
        orig_vintage = f"{commence.year}-Q{(commence.month - 1) // 3 + 1}"

        # Risk attributes
        state = np.random.choice(INDIAN_STATES, p=STATE_WEIGHTS)
        bureau_score = int(np.random.normal(700, 75))
        bureau_score = int(np.clip(bureau_score, 350, 850))

        if bureau_score >= 750:
            baseline_annual_pd = float(np.random.uniform(0.015, 0.035))
            risk_tier = "Tier 1 - Low Risk"
        elif bureau_score >= 650:
            baseline_annual_pd = float(np.random.uniform(0.040, 0.075))
            risk_tier = "Tier 2 - Medium Risk"
        else:
            baseline_annual_pd = float(np.random.uniform(0.080, 0.160))
            risk_tier = "Tier 3 - High Risk"

        contracts.append({
            "contract_id": contract_id,
            "commencement_date": commence_date,
            "origination_month": orig_month,
            "origination_vintage": orig_vintage,
            "equipment_category": cat,
            "state": state,
            "bureau_score": bureau_score,
            "risk_tier": risk_tier,
            "baseline_annual_pd": round(baseline_annual_pd, 4),
            "asset_cost": asset_cost,
            "financed_amount": financed_amount,
            "residual_value": residual_value,
            "contractual_rate": contractual_rate,
            "term_months": term_months,
            "monthly_payment": monthly_payment,
            "gross_investment": gross_investment,
            "unearned_finance_income": unearned_finance_income,
        })

        # -----------------------------------------------------
        # Calibrated Roll-Rate Delinquency State Machine
        # -----------------------------------------------------
        balance = financed_amount
        monthly_pd = baseline_annual_pd / 12.0
        dpd_status = "0_CURRENT"
        defaulted = False

        for t in range(1, term_months + 1):
            due_date = (commence + pd.DateOffset(months=t)).date()
            interest_comp = round(balance * r_monthly, 2)
            
            scheduled_cf = monthly_payment if t < term_months else round(monthly_payment + residual_value, 2)
            principal_comp = round(monthly_payment - interest_comp, 2)
            ending_balance = 0.0 if t == term_months else round(max(0.0, balance - principal_comp), 2)

            # Roll-rate logic:
            if not defaulted:
                draw = np.random.uniform(0, 1)
                
                if dpd_status == "0_CURRENT":
                    # Chance of missing payment and rolling into 30 DPD
                    if draw < monthly_pd * 2.0:
                        dpd_status = "30_DPD"
                        
                elif dpd_status == "30_DPD":
                    # 40% roll forward to 60 DPD, 45% cure back to Current, 15% stay at 30 DPD
                    if draw < 0.40:
                        dpd_status = "60_DPD"
                    elif draw < 0.85:
                        dpd_status = "0_CURRENT"
                    else:
                        dpd_status = "30_DPD"
                        
                elif dpd_status == "60_DPD":
                    # 55% roll forward to Default (terminal state), 25% cure to 30 DPD, 20% stay at 60 DPD
                    if draw < 0.55:
                        dpd_status = "90_PLUS_DEFAULT"
                        defaulted = True
                    elif draw < 0.80:
                        dpd_status = "30_DPD"
                    else:
                        dpd_status = "60_DPD"

            cashflow_rows.append({
                "contract_id": contract_id,
                "installment_num": t,
                "due_date": due_date,
                "reporting_month": due_date.strftime("%Y-%m"),
                "beginning_balance": balance,
                "interest_component": interest_comp,
                "principal_component": principal_comp,
                "expected_cashflow": scheduled_cf,
                "ending_balance": ending_balance,
                "dpd_status": dpd_status,
                "is_defaulted": 1 if defaulted else 0
            })

            balance = ending_balance

    df_contracts = pd.DataFrame(contracts)
    df_cashflows = pd.DataFrame(cashflow_rows)

    # Export to CSV
    contracts_file = OUTPUT_DIR / "contracts.csv"
    cashflows_file = OUTPUT_DIR / "monthly_cashflows.csv"

    df_contracts.to_csv(contracts_file, index=False)
    df_cashflows.to_csv(cashflows_file, index=False)

    print("\nGeneration Complete:")
    print(f" Master Contracts: {contracts_file} ({len(df_contracts):,} rows)")
    print(f" Cashflow Schedules: {cashflows_file} ({len(df_cashflows):,} rows)")
    print(f" Vintages created: {df_contracts['origination_vintage'].nunique()} distinct quarterly cohorts")

if __name__ == "__main__":
    # Increased volume to maintain cohort statistical validity across 180 months
    generate_portfolio(n_contracts=4500)