"""Visualization functions for OPM workforce analytics dashboard."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COLORS = px.colors.qualitative.Set2


def workforce_by_agency(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = df.groupby("agency")["count"].sum().nlargest(top_n).reset_index()
    fig = px.bar(data, x="count", y="agency", orientation="h",
                 title=f"Top {top_n} Agencies by Workforce Size", color_discrete_sequence=[COLORS[0]])
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig


def age_distribution(df: pd.DataFrame) -> go.Figure:
    age_order = ["LESS THAN 20", "20-24", "25-29", "30-34", "35-39", "40-44",
                 "45-49", "50-54", "55-59", "60-64", "65 OR MORE"]
    data = df.groupby("age_bracket")["count"].sum().reset_index()
    data["age_bracket"] = pd.Categorical(data["age_bracket"], categories=age_order, ordered=True)
    data = data.sort_values("age_bracket")
    fig = px.bar(data, x="age_bracket", y="count", title="Workforce Age Distribution", color_discrete_sequence=[COLORS[1]])
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def salary_by_grade(df: pd.DataFrame) -> go.Figure:
    grades = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    data = df[(df["grade"].isin(grades)) & (df["annualized_adjusted_basic_pay"] > 0)]
    fig = px.box(data, x="grade", y="annualized_adjusted_basic_pay",
                 title="Salary Distribution by Grade (GS 01-15)", color_discrete_sequence=[COLORS[2]])
    fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": grades})
    return fig


def geographic_distribution(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    data = df[df["duty_station_state"] != "REDACTED"].groupby("duty_station_state")["count"].sum().nlargest(top_n).reset_index()
    fig = px.bar(data, x="count", y="duty_station_state", orientation="h",
                 title=f"Top {top_n} States by Federal Employment", color_discrete_sequence=[COLORS[3]])
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    return fig


def accession_categories(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("accession_category")["count"].sum().reset_index()
    fig = px.pie(data, values="count", names="accession_category",
                 title="Accession Categories", color_discrete_sequence=COLORS)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def separation_categories(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("separation_category")["count"].sum().reset_index()
    fig = px.pie(data, values="count", names="separation_category",
                 title="Separation Categories", color_discrete_sequence=COLORS)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def top_hiring_agencies(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = df.groupby("agency")["count"].sum().nlargest(top_n).reset_index()
    fig = px.bar(data, x="count", y="agency", orientation="h",
                 title=f"Top {top_n} Hiring Agencies", color_discrete_sequence=[COLORS[4]])
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig


def education_distribution(df: pd.DataFrame, title: str = "Education Levels") -> go.Figure:
    edu_order = ["LESS THAN HIGH SCHOOL", "HIGH SCHOOL GRADUATE OR CERTIFICATE OF EQUIVALENCY",
                 "SOME COLLEGE - LESS THAN ONE YEAR", "SOME COLLEGE - ONE YEAR OR MORE",
                 "ASSOCIATE DEGREE", "BACHELOR'S DEGREE", "MASTER'S DEGREE",
                 "FIRST PROFESSIONAL (MD, DDS, DVM, LLB, JD)", "DOCTORATE DEGREE"]
    data = df.groupby("education_level")["count"].sum().reset_index()
    data["education_level"] = pd.Categorical(data["education_level"], categories=edu_order, ordered=True)
    fig = px.bar(data.sort_values("education_level"), x="education_level", y="count",
                 title=title, color_discrete_sequence=[COLORS[5]])
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def turnover_by_agency(accessions: pd.DataFrame, separations: pd.DataFrame, top_n: int = 15) -> go.Figure:
    acc = accessions.groupby("agency")["count"].sum().reset_index().rename(columns={"count": "accessions"})
    sep = separations.groupby("agency")["count"].sum().reset_index().rename(columns={"count": "separations"})
    merged = pd.merge(acc, sep, on="agency", how="outer").fillna(0).nlargest(top_n, "separations")
    fig = go.Figure([
        go.Bar(name="Accessions", x=merged["agency"], y=merged["accessions"], marker_color=COLORS[0]),
        go.Bar(name="Separations", x=merged["agency"], y=merged["separations"], marker_color=COLORS[1]),
    ])
    fig.update_layout(title=f"Accessions vs Separations - Top {top_n}", barmode="group", xaxis_tickangle=-45, height=500)
    return fig


def tenure_at_separation(df: pd.DataFrame) -> go.Figure:
    data = df[df["length_of_service_years"].notna()]
    return px.histogram(data, x="length_of_service_years", nbins=40,
                        title="Length of Service at Separation", color_discrete_sequence=[COLORS[6]])


def retirement_analysis(df: pd.DataFrame) -> go.Figure:
    cats = ["RETIREMENT - VOLUNTARY", "RETIREMENT - EARLY OUT", "RETIREMENT - OTHER"]
    data = df[df["separation_category"].isin(cats)].groupby("separation_category")["count"].sum().reset_index()
    return px.pie(data, values="count", names="separation_category", title="Retirement Types", color_discrete_sequence=COLORS)


def net_change_by_agency(accessions: pd.DataFrame, separations: pd.DataFrame, top_n: int = 10) -> go.Figure:
    acc = accessions.groupby("agency")["count"].sum().reset_index().rename(columns={"count": "acc"})
    sep = separations.groupby("agency")["count"].sum().reset_index().rename(columns={"count": "sep"})
    merged = pd.merge(acc, sep, on="agency", how="outer").fillna(0)
    merged["net"] = merged["acc"] - merged["sep"]
    display = pd.concat([merged.nlargest(top_n//2, "net"), merged.nsmallest(top_n//2, "net")]).drop_duplicates().sort_values("net")
    colors = [COLORS[1] if x < 0 else COLORS[0] for x in display["net"]]
    fig = px.bar(display, x="net", y="agency", orientation="h", title="Net Workforce Change by Agency")
    fig.update_traces(marker_color=colors)
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig


def kpi_card(title: str, value):
    if isinstance(value, (int, float)):
        if abs(value) >= 1_000_000:
            formatted = f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            formatted = f"{value/1_000:.0f}K"
        else:
            formatted = f"{value:,.0f}"
    else:
        formatted = value
    st.metric(label=title, value=formatted)
