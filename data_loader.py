"""Data loading utilities for OPM workforce data."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(ttl=3600)
def load_employment(sample_frac: float = 0.1) -> pd.DataFrame:
    """Load employment data from Parquet with optional sampling."""
    df = pd.read_parquet(DATA_DIR / "employment.parquet")
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    return df


@st.cache_data(ttl=3600)
def load_accessions() -> pd.DataFrame:
    """Load accessions (new hires) data from JSONL."""
    records = [json.loads(line) for line in open(DATA_DIR / "accessions.jsonl")]
    df = pd.DataFrame(records)
    df["annualized_adjusted_basic_pay"] = pd.to_numeric(df["annualized_adjusted_basic_pay"], errors="coerce")
    df["length_of_service_years"] = pd.to_numeric(df["length_of_service_years"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)
    return df


@st.cache_data(ttl=3600)
def load_separations() -> pd.DataFrame:
    """Load separations (departures) data from JSONL."""
    records = [json.loads(line) for line in open(DATA_DIR / "separations.jsonl")]
    df = pd.DataFrame(records)
    df["annualized_adjusted_basic_pay"] = pd.to_numeric(df["annualized_adjusted_basic_pay"], errors="coerce")
    df["length_of_service_years"] = pd.to_numeric(df["length_of_service_years"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)
    return df


def apply_filters(df: pd.DataFrame, agencies=None, states=None, grades=None, age_brackets=None) -> pd.DataFrame:
    """Apply sidebar filters to a DataFrame."""
    if agencies:
        df = df[df["agency"].isin(agencies)]
    if states and "duty_station_state" in df.columns:
        df = df[df["duty_station_state"].isin(states)]
    if grades and "grade" in df.columns:
        df = df[df["grade"].isin(grades)]
    if age_brackets and "age_bracket" in df.columns:
        df = df[df["age_bracket"].isin(age_brackets)]
    return df


def get_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    """Get sorted unique values from a column, excluding nulls and REDACTED."""
    values = df[column].dropna().unique().tolist()
    return sorted([v for v in values if v not in ["REDACTED", "NO DATA REPORTED"]])


def get_schema_for_ai(df: pd.DataFrame, dataset_name: str) -> str:
    """Generate a schema description for the AI."""
    lines = [f"## {dataset_name.title()} Dataset", f"Records: {len(df):,}", "### Columns"]
    for col in df.columns:
        lines.append(f"- {col} ({df[col].dtype})")
    return "\n".join(lines)


def get_sample_values(df: pd.DataFrame, column: str, n: int = 5) -> list:
    """Get sample unique values from a column for AI context."""
    return df[column].dropna().unique()[:n].tolist()
