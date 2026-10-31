from pathlib import Path
import re
import pandas as pd
from dagster import asset


def get_project_root() -> Path:
    """Resolve project root directory containing the data folder."""
    curr = Path(__file__).resolve()
    for parent in curr.parents:
        if (parent / "data").is_dir():
            return parent
    return Path.cwd().resolve()


DATA_DIR = get_project_root() / "data"


@asset(
    group_name="bronze",
    description="Raw synthetic lease contracts ingested and normalized."
)
def bronze_contracts() -> pd.DataFrame:
    """Loads lease contracts, standardizing date formats and numeric fields."""
    file_path = DATA_DIR / "synthetic_leases" / "contracts.csv"
    df = pd.read_csv(file_path)

    # Date normalization
    df["commencement_date"] = pd.to_datetime(df["commencement_date"]).dt.strftime("%Y-%m-%d")
    df["origination_month"] = pd.to_datetime(df["origination_month"]).dt.strftime("%Y-%m")

    # Numeric casting
    numeric_cols = [
        "bureau_score", "baseline_annual_pd", "asset_cost", "financed_amount",
        "residual_value", "contractual_rate", "term_months", "monthly_payment",
        "gross_investment", "unearned_finance_income"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@asset(
    group_name="bronze",
    description="Raw synthetic monthly cashflows ingested and normalized."
)
def bronze_monthly_cashflows() -> pd.DataFrame:
    """Loads monthly lease cashflows, standardizing due dates and reporting months."""
    file_path = DATA_DIR / "synthetic_leases" / "monthly_cashflows.csv"
    df = pd.read_csv(file_path)

    # Date normalization
    df["due_date"] = pd.to_datetime(df["due_date"]).dt.strftime("%Y-%m-%d")
    df["reporting_month"] = pd.to_datetime(df["reporting_month"]).dt.strftime("%Y-%m")

    # Numeric casting
    numeric_cols = [
        "installment_num", "beginning_balance", "interest_component",
        "principal_component", "expected_cashflow", "ending_balance", "is_defaulted"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@asset(
    group_name="bronze",
    description="RBI Major Monetary Policy Rates forward-filled to monthly frequency."
)
def bronze_rbi_policy_rates() -> pd.DataFrame:
    """Ingests RBI policy rates, forward-filling policy decisions into a monthly series."""
    file_path = (
        DATA_DIR
        / "rbi_monthly"
        / "Major Monetary Policy Rates and Reserve Requirements - Bank Rate, LAF (Repo, Reverse Repo, SDF and MSF) Rates, CRR & SLR.xlsx"
    )
    df = pd.read_excel(file_path)

    rename_map = {
        "Effective Date\n": "effective_date",
        "Fix Range LAF Rates (repo) ": "repo_rate",
        "Fix Range LAF Rates (reverse epo) ": "reverse_repo_rate",
        "Bank Rate": "bank_rate",
    }
    df = df.rename(columns=rename_map)[["effective_date", "repo_rate", "reverse_repo_rate", "bank_rate"]]

    # Convert non-numeric characters (hyphens, spaces) to NaN and cast to float
    for col in ["repo_rate", "reverse_repo_rate", "bank_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df = df.sort_values("effective_date").reset_index(drop=True)

    # Forward fill across change events
    df[["repo_rate", "reverse_repo_rate", "bank_rate"]] = (
        df[["repo_rate", "reverse_repo_rate", "bank_rate"]].ffill()
    )

    # Resample to monthly frequency (first day of each month) using the last rate in each month
    monthly_df = (
        df.set_index("effective_date")[["repo_rate", "reverse_repo_rate", "bank_rate"]]
        .resample("MS")
        .last()
        .ffill()
    )
    monthly_df["reporting_month"] = monthly_df.index.strftime("%Y-%m")
    result = monthly_df.reset_index(drop=True)[["reporting_month", "repo_rate", "reverse_repo_rate", "bank_rate"]]
    return result


@asset(
    group_name="bronze",
    description="RBI Weighted Average Call Money Rates melted to monthly long format."
)
def bronze_rbi_call_money() -> pd.DataFrame:
    """Melts weighted average call money rates across fiscal years into monthly records."""
    file_path = DATA_DIR / "rbi_monthly" / "Weighted Average Call Money Rates.xlsx"
    df = pd.read_excel(file_path)

    # Clean header whitespace
    df.columns = [c.strip() for c in df.columns]

    # Skip numbering row if present
    if str(df.iloc[0]["Year"]).strip() in ["1", "1.0"]:
        df = df.iloc[1:].copy()

    df["Year"] = df["Year"].ffill()
    df_avg = df[df["Range"].astype(str).str.strip().str.lower() == "average"].copy()

    # Month offset mapping for Indian Fiscal Year (April = Year, Jan-March = Year + 1)
    month_mapping = {
        "Apr.": ("04", 0),
        "May": ("05", 0),
        "Jun.": ("06", 0),
        "Jul.": ("07", 0),
        "Aug.": ("08", 0),
        "Sep.": ("09", 0),
        "Oct.": ("10", 0),
        "Nov.": ("11", 0),
        "Dec.": ("12", 0),
        "Jan.": ("01", 1),
        "Feb.": ("02", 1),
        "Mar.": ("03", 1),
    }

    records = []
    for _, row in df_avg.iterrows():
        year_str = str(row["Year"]).strip()
        if "-" not in year_str:
            continue
        start_year = int(year_str.split("-")[0])
        for col, (m_str, yr_offset) in month_mapping.items():
            if col in row:
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val):
                    yr = start_year + yr_offset
                    records.append({
                        "reporting_month": f"{yr:04d}-{m_str}",
                        "call_money_rate": float(val),
                    })

    result = pd.DataFrame(records).sort_values("reporting_month").reset_index(drop=True)
    return result


@asset(
    group_name="bronze",
    description="RBI secondary market T-Bill yields for residual maturities."
)
def bronze_rbi_tbill_yields() -> pd.DataFrame:
    """Standardizes secondary market T-Bill yields across residual maturity buckets."""
    file_path = (
        DATA_DIR
        / "rbi_monthly"
        / "Yield of SGL Transactions in Treasury Bills for Residual Maturities.xlsx"
    )
    df = pd.read_excel(file_path)

    rename_map = {
        "Year/Month": "reporting_month",
        "Upto 14 days": "tbill_yield_14d",
        "15-91 days": "tbill_yield_91d",
        "92-182 days": "tbill_yield_182d",
        "183-364 days": "tbill_yield_364d",
    }
    df = df.rename(columns=rename_map)

    # Convert MMM-YYYY (e.g. Jun-2026) to YYYY-MM
    df["reporting_month"] = pd.to_datetime(df["reporting_month"], format="%b-%Y").dt.strftime("%Y-%m")

    # Cast yield columns to float, coercing hyphens and blanks to NaN
    yield_cols = ["tbill_yield_14d", "tbill_yield_91d", "tbill_yield_182d", "tbill_yield_364d"]
    for col in yield_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df[["reporting_month"] + yield_cols].sort_values("reporting_month").reset_index(drop=True)
    return result


@asset(
    group_name="bronze",
    description="RBI Deployment of Bank Credit by Major Sectors (including NBFCs) in long format."
)
def bronze_rbi_nbfc_credit() -> pd.DataFrame:
    """Ingests sectoral bank credit from data/rbi_bank_credit/bank_cr_major_sectors.xlsx.

    Normalizes sector names and computes 12-month YoY growth percentages into long format:
    ['reporting_month', 'sector_name', 'outstanding_inr_crore', 'yoy_growth_pct'].
    """
    file_path = DATA_DIR / "rbi_bank_credit" / "bank_cr_major_sectors.xlsx"
    df = pd.read_excel(file_path)

    # Extract reporting months from row 0
    dates_map = {}
    for c in range(2, df.shape[1]):
        val = df.iloc[0, c]
        if pd.notna(val):
            try:
                m_str = pd.to_datetime(val).strftime("%Y-%m")
                dates_map[c] = m_str
            except Exception:
                pass

    records = []
    for r in range(1, len(df)):
        sec = df.iloc[r, 1]
        if pd.isna(sec) or not str(sec).strip():
            continue

        # Clean sector name: remove footnotes, superscripts, extra spaces, trailing commas/of which
        sec_clean = str(sec).strip().replace("\n", " ")
        sec_clean = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹*]", "", sec_clean)
        sec_clean = re.sub(r"\s+of which[,]?$", "", sec_clean, flags=re.IGNORECASE)
        sec_clean = re.sub(r"\s+", " ", sec_clean).strip()

        for c, m_str in dates_map.items():
            val = df.iloc[r, c]
            val_num = pd.to_numeric(val, errors="coerce")
            if pd.notna(val_num):
                records.append({
                    "reporting_month": m_str,
                    "sector_name": sec_clean,
                    "outstanding_inr_crore": float(val_num),
                })

    res = pd.DataFrame(records)

    # Sort to compute 12-month YoY percentage change
    res = res.sort_values(["sector_name", "reporting_month"]).reset_index(drop=True)
    res["yoy_growth_pct"] = (
        res.groupby("sector_name")["outstanding_inr_crore"].pct_change(periods=12) * 100.0
    )

    result = res[["reporting_month", "sector_name", "outstanding_inr_crore", "yoy_growth_pct"]]
    return result.sort_values(["reporting_month", "sector_name"]).reset_index(drop=True)
