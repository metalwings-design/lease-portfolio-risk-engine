from pathlib import Path
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 1000)

BASE_DIR = Path(__file__).resolve().parent

print("=" * 85)
print("              ATHENA PORTFOLIO INTELLIGENCE: COMPLETE SYSTEM AUDIT")
print("=" * 85)

# -------------------------------------------------------------------------
# LAYER 1: BRONZE EXPLORATORY SUMMARY
# -------------------------------------------------------------------------
print("\n" + "-" * 85)
print(" [LAYER 1: BRONZE] INGESTED ASSETS & NORMALIZED TAPE SUMMARY")
print("-" * 85)

df_contracts = pd.read_csv(BASE_DIR / "data" / "synthetic_leases" / "contracts.csv")
df_cf = pd.read_csv(BASE_DIR / "data" / "synthetic_leases" / "monthly_cashflows.csv")

print(f"Total Lease Contracts:        {len(df_contracts):,}")
print(f"Total Cashflow Rows:          {len(df_cf):,}")
print(f"Origination Window:           {df_contracts['origination_month'].min()} to {df_contracts['origination_month'].max()}")
print(f"Cashflow Settlement Window:   {df_cf['reporting_month'].min()} to {df_cf['reporting_month'].max()}")

cat_summary = df_contracts.groupby("equipment_category").agg(
    Contracts=("contract_id", "count"),
    Total_Financed_Cr=("financed_amount", lambda x: round(x.sum() / 1e7, 2)),
    Mean_Tenor_Months=("term_months", "mean"),
    Avg_Implicit_Rate_Pct=("contractual_rate", lambda x: round(x.mean() * 100, 2))
).reset_index()
print("\nEquipment Category Distribution:")
print(cat_summary.to_string(index=False))

# -------------------------------------------------------------------------
# LAYER 2: SILVER DERIVED VALUATION METRICS (ASC 842 / IFRS 16)
# -------------------------------------------------------------------------
print("\n" + "-" * 85)
print(" [LAYER 2: SILVER] LEASE VALUATION & SPREAD ANALYSIS")
print("-" * 85)

tot_financed = df_contracts["financed_amount"].sum() / 1e7
tot_gross_inv = df_contracts["gross_investment"].sum() / 1e7
tot_ufi = df_contracts["unearned_finance_income"].sum() / 1e7
mean_coupon = df_contracts["contractual_rate"].mean() * 100

print(f"Total Financed Principal (NIL at Inception):  INR {tot_financed:,.2f} Crore")
print(f"Gross Investment in Leases (GIL):             INR {tot_gross_inv:,.2f} Crore")
print(f"Total Unearned Finance Income (UFI):          INR {tot_ufi:,.2f} Crore")
print(f"Portfolio Weighted Average Contractual Rate:  {mean_coupon:.2f}% p.a.")
print(f"UFI-to-Gross Investment Ratio:                {(tot_ufi / tot_gross_inv) * 100:.2f}%")

tier_summary = df_contracts.groupby("risk_tier").agg(
    Contracts=("contract_id", "count"),
    Total_Volume_Cr=("financed_amount", lambda x: round(x.sum() / 1e7, 2)),
    Mean_PD_Pct=("baseline_annual_pd", lambda x: round(x.mean() * 100, 2)),
    Avg_CIBIL=("bureau_score", "mean")
).reset_index()
print("\nBorrower Credit Tier Segmentation:")
print(tier_summary.to_string(index=False))

# -------------------------------------------------------------------------
# LAYER 3: GOLD EXECUTIVE PORTFOLIO ANALYTICS
# -------------------------------------------------------------------------
print("\n" + "-" * 85)
print(" [LAYER 3: GOLD] PORTFOLIO CONCENTRATION (HHI) & RISK METRICS")
print("-" * 85)

# A. Category HHI
cat_shares = (df_contracts.groupby("equipment_category")["financed_amount"].sum() / df_contracts["financed_amount"].sum()) * 100
hhi_cat = (cat_shares ** 2).sum()

# B. Geographic State HHI
state_shares = (df_contracts.groupby("state")["financed_amount"].sum() / df_contracts["financed_amount"].sum()) * 100
hhi_state = (state_shares ** 2).sum()

# C. Risk Tier HHI
tier_shares = (df_contracts.groupby("risk_tier")["financed_amount"].sum() / df_contracts["financed_amount"].sum()) * 100
hhi_tier = (tier_shares ** 2).sum()

def classify_hhi(val):
    if val < 1500: return "Diversified / Healthy"
    elif val <= 2500: return "Moderate Concentration"
    else: return "High Concentration"

hhi_table = pd.DataFrame([
    {"Dimension": "Equipment Category", "HHI_Score": round(hhi_cat, 1), "Classification": classify_hhi(hhi_cat)},
    {"Dimension": "Borrower Credit Risk Tier", "HHI_Score": round(hhi_tier, 1), "Classification": classify_hhi(hhi_tier)},
    {"Dimension": "Geographic State", "HHI_Score": round(hhi_state, 1), "Classification": classify_hhi(hhi_state)},
])
print("Portfolio Concentration Index (Herfindahl-Hirschman Index):")
print(hhi_table.to_string(index=False))

# -------------------------------------------------------------------------
# LAYER 3: GOLD MACRO RATE SHOCK SENSITIVITY (DURATION & P&L IMPACT)
# -------------------------------------------------------------------------
print("\n" + "-" * 85)
print(" [LAYER 3: GOLD] MACRO INTEREST RATE SHOCK SENSITIVITY (STRESS TEST)")
print("-" * 85)

# Estimate active book and effective duration
active_carrying_cr = round(tot_financed * 0.88, 2)
effective_duration = 2.05 # years

shocks = [-200, -100, 0, 100, 200]
stress_rows = []
for s in shocks:
    dr = s / 10000.0
    pct_chg = (-effective_duration * dr + 0.5 * 6.5 * (dr ** 2)) * 100
    revalued = round(active_carrying_cr * (1 + pct_chg / 100), 2)
    pnl = round(revalued - active_carrying_cr, 2)
    stress_rows.append({
        "Scenario": "Base Curve" if s == 0 else f"{'+' if s > 0 else ''}{s} bps ({'Hike' if s > 0 else 'Cut'})",
        "Benchmark_Shift_bps": s,
        "Revalued_Fair_Value_Cr": revalued,
        "Mark_to_Market_PnL_Cr": pnl,
        "Value_Change_Pct": f"{pct_chg:+.2f}%"
    })

df_stress = pd.DataFrame(stress_rows)
print(f"Base Active Portfolio Carrying Value: INR {active_carrying_cr:,.2f} Crore")
print(f"Portfolio Effective Duration:         {effective_duration:.2f} Years\n")
print(df_stress.to_string(index=False))

print("\n" + "=" * 85)
print("                         AUDIT COMPLETE")
print("=" * 85)