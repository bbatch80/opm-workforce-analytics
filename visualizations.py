"""Visualization functions for OPM workforce analytics dashboard."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Consistent color palette
COLORS = px.colors.qualitative.Set2


def workforce_by_agency(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of workforce by agency (top N)."""
    agency_counts = (
        df.groupby("agency")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        agency_counts,
        x="count",
        y="agency",
        orientation="h",
        title=f"Top {top_n} Agencies by Workforce Size",
        labels={"count": "Employees", "agency": "Agency"},
        color_discrete_sequence=[COLORS[0]],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig


def age_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of workforce age distribution."""
    # Define age bracket order
    age_order = [
        "LESS THAN 20",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65 OR MORE",
    ]

    age_counts = df.groupby("age_bracket")["count"].sum().reset_index()
    age_counts["age_bracket"] = pd.Categorical(
        age_counts["age_bracket"], categories=age_order, ordered=True
    )
    age_counts = age_counts.sort_values("age_bracket")

    fig = px.bar(
        age_counts,
        x="age_bracket",
        y="count",
        title="Workforce Age Distribution",
        labels={"count": "Employees", "age_bracket": "Age Bracket"},
        color_discrete_sequence=[COLORS[1]],
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def salary_by_grade(df: pd.DataFrame) -> go.Figure:
    """Box plot of salary distribution by grade."""
    # Filter to common grades and remove nulls
    common_grades = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    df_filtered = df[
        (df["grade"].isin(common_grades))
        & (df["annualized_adjusted_basic_pay"].notna())
        & (df["annualized_adjusted_basic_pay"] > 0)
    ]

    fig = px.box(
        df_filtered,
        x="grade",
        y="annualized_adjusted_basic_pay",
        title="Salary Distribution by Grade (GS 01-15)",
        labels={"annualized_adjusted_basic_pay": "Annual Salary ($)", "grade": "Grade"},
        color_discrete_sequence=[COLORS[2]],
    )
    fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": common_grades})
    return fig


def geographic_distribution(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of workforce by state."""
    state_counts = (
        df[df["duty_station_state"] != "REDACTED"]
        .groupby("duty_station_state")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        state_counts,
        x="count",
        y="duty_station_state",
        orientation="h",
        title=f"Top {top_n} States by Federal Employment",
        labels={"count": "Employees", "duty_station_state": "State"},
        color_discrete_sequence=[COLORS[3]],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    return fig


def accession_categories(df: pd.DataFrame) -> go.Figure:
    """Pie chart of accession categories."""
    cat_counts = df.groupby("accession_category")["count"].sum().reset_index()

    fig = px.pie(
        cat_counts,
        values="count",
        names="accession_category",
        title="Accession Categories (How People Joined)",
        color_discrete_sequence=COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def separation_categories(df: pd.DataFrame) -> go.Figure:
    """Pie chart of separation categories."""
    cat_counts = df.groupby("separation_category")["count"].sum().reset_index()

    fig = px.pie(
        cat_counts,
        values="count",
        names="separation_category",
        title="Separation Categories (Why People Left)",
        color_discrete_sequence=COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def top_hiring_agencies(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of top hiring agencies."""
    agency_counts = (
        df.groupby("agency")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        agency_counts,
        x="count",
        y="agency",
        orientation="h",
        title=f"Top {top_n} Hiring Agencies",
        labels={"count": "New Hires", "agency": "Agency"},
        color_discrete_sequence=[COLORS[4]],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig


def education_distribution(df: pd.DataFrame, title: str = "Education Levels") -> go.Figure:
    """Bar chart of education levels."""
    edu_order = [
        "LESS THAN HIGH SCHOOL",
        "HIGH SCHOOL GRADUATE OR CERTIFICATE OF EQUIVALENCY",
        "SOME COLLEGE - LESS THAN ONE YEAR",
        "SOME COLLEGE - ONE YEAR OR MORE",
        "ASSOCIATE DEGREE",
        "BACHELOR'S DEGREE",
        "MASTER'S DEGREE",
        "FIRST PROFESSIONAL (MD, DDS, DVM, LLB, JD)",
        "DOCTORATE DEGREE",
    ]

    edu_counts = df.groupby("education_level")["count"].sum().reset_index()
    edu_counts["education_level"] = pd.Categorical(
        edu_counts["education_level"], categories=edu_order, ordered=True
    )
    edu_counts = edu_counts.sort_values("education_level")

    fig = px.bar(
        edu_counts,
        x="education_level",
        y="count",
        title=title,
        labels={"count": "Employees", "education_level": "Education Level"},
        color_discrete_sequence=[COLORS[5]],
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def turnover_by_agency(accessions: pd.DataFrame, separations: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart comparing accessions and separations by agency."""
    acc_counts = accessions.groupby("agency")["count"].sum().reset_index()
    acc_counts.columns = ["agency", "accessions"]

    sep_counts = separations.groupby("agency")["count"].sum().reset_index()
    sep_counts.columns = ["agency", "separations"]

    merged = pd.merge(acc_counts, sep_counts, on="agency", how="outer").fillna(0)
    merged["net_change"] = merged["accessions"] - merged["separations"]
    merged = merged.nlargest(top_n, "separations")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Accessions (In)",
            x=merged["agency"],
            y=merged["accessions"],
            marker_color=COLORS[0],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Separations (Out)",
            x=merged["agency"],
            y=merged["separations"],
            marker_color=COLORS[1],
        )
    )

    fig.update_layout(
        title=f"Accessions vs Separations - Top {top_n} Agencies",
        barmode="group",
        xaxis_tickangle=-45,
        height=500,
    )
    return fig


def tenure_at_separation(df: pd.DataFrame) -> go.Figure:
    """Histogram of length of service at separation."""
    df_filtered = df[df["length_of_service_years"].notna()]

    fig = px.histogram(
        df_filtered,
        x="length_of_service_years",
        nbins=40,
        title="Length of Service at Separation",
        labels={"length_of_service_years": "Years of Service", "count": "Employees"},
        color_discrete_sequence=[COLORS[6]],
    )
    return fig


def retirement_analysis(df: pd.DataFrame) -> go.Figure:
    """Breakdown of retirement types."""
    retirement_cats = [
        "RETIREMENT - VOLUNTARY",
        "RETIREMENT - EARLY OUT",
        "RETIREMENT - OTHER",
    ]
    df_retirement = df[df["separation_category"].isin(retirement_cats)]

    fig = px.pie(
        df_retirement.groupby("separation_category")["count"].sum().reset_index(),
        values="count",
        names="separation_category",
        title="Retirement Types",
        color_discrete_sequence=COLORS,
    )
    return fig


def net_change_by_agency(accessions: pd.DataFrame, separations: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Waterfall-style chart showing net workforce change by agency."""
    acc_counts = accessions.groupby("agency")["count"].sum().reset_index()
    acc_counts.columns = ["agency", "accessions"]

    sep_counts = separations.groupby("agency")["count"].sum().reset_index()
    sep_counts.columns = ["agency", "separations"]

    merged = pd.merge(acc_counts, sep_counts, on="agency", how="outer").fillna(0)
    merged["net_change"] = merged["accessions"] - merged["separations"]

    # Get agencies with biggest changes (positive and negative)
    top_gainers = merged.nlargest(top_n // 2, "net_change")
    top_losers = merged.nsmallest(top_n // 2, "net_change")
    display_df = pd.concat([top_gainers, top_losers]).drop_duplicates()
    display_df = display_df.sort_values("net_change")

    colors = [COLORS[1] if x < 0 else COLORS[0] for x in display_df["net_change"]]

    fig = px.bar(
        display_df,
        x="net_change",
        y="agency",
        orientation="h",
        title="Net Workforce Change by Agency",
        labels={"net_change": "Net Change", "agency": "Agency"},
    )
    fig.update_traces(marker_color=colors)
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig


def kpi_card(title: str, value: str | int | float, delta: str | None = None) -> None:
    """Display a KPI metric card using Streamlit."""
    import streamlit as st

    if isinstance(value, (int, float)):
        if abs(value) >= 1_000_000:
            formatted = f"{value / 1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            formatted = f"{value / 1_000:.0f}K"
        else:
            formatted = f"{value:,.0f}"
    else:
        formatted = value

    st.metric(label=title, value=formatted, delta=delta)
