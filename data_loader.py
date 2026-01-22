"""Data loading utilities with caching and sampling for OPM workforce data."""
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

# Column metadata for schema export
COLUMN_DESCRIPTIONS = {
    "age_bracket": "Age range (20-24, 25-29, 30-34, etc.)",
    "agency": "Parent department/agency name",
    "agency_code": "2-letter agency code (VA, HS, TR, DD)",
    "agency_subelement": "Sub-agency or bureau name",
    "agency_subelement_code": "Sub-agency code",
    "annualized_adjusted_basic_pay": "Annual salary (numeric, may have nulls for REDACTED)",
    "appointment_type": "Employment status type (CAREER, CAREER-CONDITIONAL, etc.)",
    "count": "Number of employees (usually 1)",
    "duty_station_country": "Work country",
    "duty_station_state": "Work state",
    "education_level": "Highest education (HIGH SCHOOL, BACHELOR'S, MASTER'S, etc.)",
    "grade": "Pay grade (01-15, SES, etc.)",
    "length_of_service_years": "Federal tenure in years",
    "occupational_group": "Job category (MEDICAL, INVESTIGATION, LEGAL)",
    "occupational_series": "Specific job type (NURSE, POLICE, IT MANAGEMENT)",
    "pay_plan": "Pay system (GENERAL SCHEDULE, FEDERAL WAGE SYSTEM)",
    "stem_occupation": "STEM classification (STEM OCCUPATIONS, HEALTH, ALL OTHER)",
    "supervisory_status": "Management level (ALL OTHER, SUPERVISOR, MANAGER)",
    "work_schedule": "Work type (FULL-TIME, PART-TIME, INTERMITTENT)",
    "snapshot_yyyymm": "Data snapshot date (employment only)",
    "accession_category": "Type of hire (accessions only)",
    "separation_category": "Reason for leaving (separations only)",
    "personnel_action_effective_date_yyyymm": "Action date (accessions/separations)",
}


@st.cache_data(ttl=3600)
def load_employment(sample_frac: float = 0.1) -> pd.DataFrame:
    """Load employment data from Parquet with optional sampling.

    Args:
        sample_frac: Fraction of data to sample (0.1 = 10%). Set to 1.0 for full data.

    Returns:
        DataFrame with employment records.
    """
    parquet_path = DATA_DIR / "employment.parquet"
    df = pd.read_parquet(parquet_path)

    if sample_frac < 1.0:
        # Simple random sample (stratified adds complexity for minimal benefit here)
        df = df.sample(frac=sample_frac, random_state=42)

    return df


@st.cache_data(ttl=3600)
def load_accessions() -> pd.DataFrame:
    """Load accessions (new hires) data from JSONL."""
    jsonl_path = DATA_DIR / "accessions.jsonl"
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Convert numeric columns
    df["annualized_adjusted_basic_pay"] = pd.to_numeric(
        df["annualized_adjusted_basic_pay"], errors="coerce"
    )
    df["length_of_service_years"] = pd.to_numeric(
        df["length_of_service_years"], errors="coerce"
    )
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)

    return df


@st.cache_data(ttl=3600)
def load_separations() -> pd.DataFrame:
    """Load separations (departures) data from JSONL."""
    jsonl_path = DATA_DIR / "separations.jsonl"
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Convert numeric columns
    df["annualized_adjusted_basic_pay"] = pd.to_numeric(
        df["annualized_adjusted_basic_pay"], errors="coerce"
    )
    df["length_of_service_years"] = pd.to_numeric(
        df["length_of_service_years"], errors="coerce"
    )
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)

    return df


def apply_filters(
    df: pd.DataFrame,
    agencies: Optional[list[str]] = None,
    states: Optional[list[str]] = None,
    grades: Optional[list[str]] = None,
    age_brackets: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Apply sidebar filters to a DataFrame.

    Args:
        df: Source DataFrame
        agencies: List of agency names to include
        states: List of states to include
        grades: List of grades to include
        age_brackets: List of age brackets to include

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    if agencies:
        filtered = filtered[filtered["agency"].isin(agencies)]

    if states and "duty_station_state" in filtered.columns:
        filtered = filtered[filtered["duty_station_state"].isin(states)]

    if grades and "grade" in filtered.columns:
        filtered = filtered[filtered["grade"].isin(grades)]

    if age_brackets and "age_bracket" in filtered.columns:
        filtered = filtered[filtered["age_bracket"].isin(age_brackets)]

    return filtered


def get_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    """Get sorted unique values from a column, excluding nulls and REDACTED."""
    values = df[column].dropna().unique().tolist()
    values = [v for v in values if v not in ["REDACTED", "NO DATA REPORTED"]]
    return sorted(values)


def get_schema_for_ai(df: pd.DataFrame, dataset_name: str) -> str:
    """Generate a schema description for the AI to understand the data.

    Args:
        df: DataFrame to describe
        dataset_name: Name of the dataset (employment, accessions, separations)

    Returns:
        Markdown-formatted schema description
    """
    lines = [f"## {dataset_name.title()} Dataset"]
    lines.append(f"- Records: {len(df):,}")
    lines.append(f"- Columns: {len(df.columns)}")
    lines.append("")
    lines.append("### Columns")
    lines.append("| Column | Type | Non-Null | Description |")
    lines.append("|--------|------|----------|-------------|")

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        desc = COLUMN_DESCRIPTIONS.get(col, "")
        lines.append(f"| {col} | {dtype} | {non_null:,} | {desc} |")

    return "\n".join(lines)


def get_sample_values(df: pd.DataFrame, column: str, n: int = 5) -> list:
    """Get sample unique values from a column for AI context."""
    values = df[column].dropna().unique()[:n].tolist()
    return values


@st.cache_data(ttl=3600)
def get_summary_stats() -> dict:
    """Calculate summary statistics for KPI cards.

    Returns:
        Dictionary with workforce statistics.
    """
    employment = load_employment(sample_frac=1.0)
    accessions = load_accessions()
    separations = load_separations()

    # Sum counts for aggregated data
    total_employees = employment["count"].sum() if "count" in employment.columns else len(employment)
    total_accessions = accessions["count"].sum() if "count" in accessions.columns else len(accessions)
    total_separations = separations["count"].sum() if "count" in separations.columns else len(separations)

    return {
        "total_employees": int(total_employees),
        "total_accessions": int(total_accessions),
        "total_separations": int(total_separations),
        "net_change": int(total_accessions - total_separations),
        "avg_salary_employment": employment["annualized_adjusted_basic_pay"].mean(),
        "avg_salary_accessions": accessions["annualized_adjusted_basic_pay"].mean(),
        "avg_salary_separations": separations["annualized_adjusted_basic_pay"].mean(),
        "num_agencies": employment["agency"].nunique(),
    }
