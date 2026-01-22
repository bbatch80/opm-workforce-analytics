"""OPM Workforce Analytics Dashboard."""
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import plotly.express as px

from data_loader import load_employment, load_accessions, load_separations, apply_filters, get_unique_values
from visualizations import (workforce_by_agency, age_distribution, salary_by_grade, geographic_distribution,
                            accession_categories, separation_categories, top_hiring_agencies, education_distribution,
                            turnover_by_agency, tenure_at_separation, retirement_analysis, net_change_by_agency, kpi_card)
from query import query_ai, EXAMPLE_QUERIES

st.set_page_config(page_title="OPM Workforce Analytics", page_icon="📊", layout="wide")
st.title("Federal Workforce Analytics Dashboard")
st.caption("November 2025 OPM Data | 2M+ employees, 13K accessions, 21K separations")

# Load data
employment = load_employment(sample_frac=0.1)
accessions = load_accessions()
separations = load_separations()

# Sidebar filters
st.sidebar.header("Filters")
all_agencies = sorted(set(get_unique_values(employment, "agency") + get_unique_values(accessions, "agency")))
selected_agencies = st.sidebar.multiselect("Agency", all_agencies, placeholder="All agencies")
selected_states = st.sidebar.multiselect("State", get_unique_values(employment, "duty_station_state"), placeholder="All states")
selected_grades = st.sidebar.multiselect("Grade", get_unique_values(employment, "grade"), placeholder="All grades")

# Apply filters
filtered_emp = apply_filters(employment, agencies=selected_agencies or None, states=selected_states or None, grades=selected_grades or None)
filtered_acc = apply_filters(accessions, agencies=selected_agencies or None, states=selected_states or None, grades=selected_grades or None)
filtered_sep = apply_filters(separations, agencies=selected_agencies or None, states=selected_states or None, grades=selected_grades or None)

# Tabs
tab_overview, tab_hiring, tab_turnover, tab_ai = st.tabs(["Overview", "Hiring", "Turnover", "AI Analyst"])

# OVERVIEW TAB
with tab_overview:
    st.header("Workforce Overview")
    c1, c2, c3, c4 = st.columns(4)
    total_emp, total_acc, total_sep = filtered_emp["count"].sum(), filtered_acc["count"].sum(), filtered_sep["count"].sum()
    with c1: kpi_card("Total Employees", total_emp)
    with c2: kpi_card("Accessions", total_acc)
    with c3: kpi_card("Separations", total_sep)
    with c4: st.metric("Net Change", f"{total_acc - total_sep:+,}")
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(workforce_by_agency(filtered_emp), use_container_width=True)
    with c2: st.plotly_chart(age_distribution(filtered_emp), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(salary_by_grade(filtered_emp), use_container_width=True)
    with c2: st.plotly_chart(geographic_distribution(filtered_emp), use_container_width=True)

# HIRING TAB
with tab_hiring:
    st.header("Hiring Analysis")
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Total Accessions", filtered_acc["count"].sum())
    with c2: st.metric("Avg Salary", f"${filtered_acc['annualized_adjusted_basic_pay'].mean():,.0f}")
    with c3: kpi_card("Agencies Hiring", filtered_acc["agency"].nunique())
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(accession_categories(filtered_acc), use_container_width=True)
    with c2: st.plotly_chart(top_hiring_agencies(filtered_acc), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(education_distribution(filtered_acc, "Education of New Hires"), use_container_width=True)
    with c2:
        state_data = filtered_acc[filtered_acc["duty_station_state"] != "REDACTED"].groupby("duty_station_state")["count"].sum().nlargest(15).reset_index()
        fig = px.bar(state_data, x="count", y="duty_station_state", orientation="h", title="Top States for Hiring")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# TURNOVER TAB
with tab_turnover:
    st.header("Turnover Analysis")
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Total Separations", filtered_sep["count"].sum())
    with c2: st.metric("Avg Tenure", f"{filtered_sep['length_of_service_years'].mean():.1f} yrs")
    with c3:
        ret_pct = filtered_sep[filtered_sep["separation_category"].str.contains("RETIREMENT", na=False)]["count"].sum() / filtered_sep["count"].sum() * 100
        st.metric("Retirement %", f"{ret_pct:.1f}%")
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(separation_categories(filtered_sep), use_container_width=True)
    with c2: st.plotly_chart(turnover_by_agency(filtered_acc, filtered_sep), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(tenure_at_separation(filtered_sep), use_container_width=True)
    with c2: st.plotly_chart(retirement_analysis(filtered_sep), use_container_width=True)
    st.plotly_chart(net_change_by_agency(filtered_acc, filtered_sep), use_container_width=True)

# AI ANALYST TAB
with tab_ai:
    st.header("AI Analyst")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try: api_key = st.secrets.get("OPENAI_API_KEY", "")
        except: pass

    if not api_key:
        st.warning("Set OPENAI_API_KEY in environment or .streamlit/secrets.toml")
    else:
        st.subheader("Example Questions")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUERIES):
            if cols[i % 2].button(q, key=f"ex_{i}"):
                st.session_state["question"] = q

        question = st.text_input("Ask a question", value=st.session_state.get("question", ""))
        if st.button("Analyze", type="primary", disabled=not question):
            with st.spinner("Analyzing..."):
                try:
                    result = query_ai(question, filtered_emp, filtered_acc, filtered_sep)
                    st.subheader("Answer")
                    st.write(result.explanation)
                    if result.execution.figure:
                        st.plotly_chart(result.execution.figure, use_container_width=True)
                    if result.execution.dataframe is not None:
                        st.dataframe(result.execution.dataframe.head(20))
                    with st.expander("View code"):
                        st.code(result.code, language="python")
                    if not result.execution.success:
                        st.error(result.execution.error)
                except Exception as e:
                    st.error(f"Error: {e}")

st.sidebar.divider()
st.sidebar.caption("[OPM FedScope](https://www.opm.gov/data/Index.aspx) | Built with Streamlit")
