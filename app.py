"""OPM Workforce Analytics Dashboard - Main Streamlit Application."""
import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file if it exists

import streamlit as st

from data_loader import (
    load_employment,
    load_accessions,
    load_separations,
    apply_filters,
    get_unique_values,
    get_summary_stats,
)
from visualizations import (
    workforce_by_agency,
    age_distribution,
    salary_by_grade,
    geographic_distribution,
    accession_categories,
    separation_categories,
    top_hiring_agencies,
    education_distribution,
    turnover_by_agency,
    tenure_at_separation,
    retirement_analysis,
    net_change_by_agency,
    kpi_card,
)
from query import query_ai, EXAMPLE_QUERIES

# Page configuration
st.set_page_config(
    page_title="OPM Workforce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("Federal Workforce Analytics Dashboard")
st.caption("November 2025 OPM Data | 2M+ employees, 13K accessions, 21K separations")


# Load data with caching
@st.cache_data(show_spinner="Loading employment data...")
def get_employment():
    return load_employment(sample_frac=0.1)  # 10% sample for interactive use


@st.cache_data(show_spinner="Loading accessions data...")
def get_accessions():
    return load_accessions()


@st.cache_data(show_spinner="Loading separations data...")
def get_separations():
    return load_separations()


# Load data
employment = get_employment()
accessions = get_accessions()
separations = get_separations()

# Sidebar filters
st.sidebar.header("Filters")

# Agency filter
all_agencies = sorted(set(
    list(get_unique_values(employment, "agency")) +
    list(get_unique_values(accessions, "agency")) +
    list(get_unique_values(separations, "agency"))
))
selected_agencies = st.sidebar.multiselect(
    "Agency",
    options=all_agencies,
    default=[],
    placeholder="All agencies",
)

# State filter
all_states = get_unique_values(employment, "duty_station_state")
selected_states = st.sidebar.multiselect(
    "State",
    options=all_states,
    default=[],
    placeholder="All states",
)

# Grade filter
all_grades = get_unique_values(employment, "grade")
selected_grades = st.sidebar.multiselect(
    "Grade",
    options=all_grades,
    default=[],
    placeholder="All grades",
)

# Age bracket filter
all_age_brackets = get_unique_values(employment, "age_bracket")
selected_age_brackets = st.sidebar.multiselect(
    "Age Bracket",
    options=all_age_brackets,
    default=[],
    placeholder="All ages",
)

# Apply filters
filtered_employment = apply_filters(
    employment,
    agencies=selected_agencies or None,
    states=selected_states or None,
    grades=selected_grades or None,
    age_brackets=selected_age_brackets or None,
)
filtered_accessions = apply_filters(
    accessions,
    agencies=selected_agencies or None,
    states=selected_states or None,
    grades=selected_grades or None,
    age_brackets=selected_age_brackets or None,
)
filtered_separations = apply_filters(
    separations,
    agencies=selected_agencies or None,
    states=selected_states or None,
    grades=selected_grades or None,
    age_brackets=selected_age_brackets or None,
)

# Show filter status
if any([selected_agencies, selected_states, selected_grades, selected_age_brackets]):
    st.sidebar.info(
        f"Showing {len(filtered_employment):,} employees, "
        f"{len(filtered_accessions):,} accessions, "
        f"{len(filtered_separations):,} separations"
    )

# Tabs
tab_overview, tab_hiring, tab_turnover, tab_ai = st.tabs([
    "Overview",
    "Hiring (Accessions)",
    "Turnover (Separations)",
    "AI Analyst",
])

# =============================================================================
# OVERVIEW TAB
# =============================================================================
with tab_overview:
    st.header("Workforce Overview")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    total_emp = filtered_employment["count"].sum()
    total_acc = filtered_accessions["count"].sum()
    total_sep = filtered_separations["count"].sum()
    net_change = total_acc - total_sep

    with col1:
        kpi_card("Total Employees", total_emp)
    with col2:
        kpi_card("Accessions", total_acc)
    with col3:
        kpi_card("Separations", total_sep)
    with col4:
        delta_str = f"+{net_change:,}" if net_change > 0 else f"{net_change:,}"
        st.metric("Net Change", delta_str)

    st.divider()

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            workforce_by_agency(filtered_employment),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            age_distribution(filtered_employment),
            use_container_width=True,
        )

    # Charts row 2
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            salary_by_grade(filtered_employment),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            geographic_distribution(filtered_employment),
            use_container_width=True,
        )

# =============================================================================
# HIRING TAB
# =============================================================================
with tab_hiring:
    st.header("Hiring Analysis (Accessions)")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Total Accessions", filtered_accessions["count"].sum())
    with col2:
        avg_salary = filtered_accessions["annualized_adjusted_basic_pay"].mean()
        st.metric("Avg Salary (New Hires)", f"${avg_salary:,.0f}" if avg_salary else "N/A")
    with col3:
        num_agencies = filtered_accessions["agency"].nunique()
        kpi_card("Agencies Hiring", num_agencies)
    with col4:
        new_hire_pct = (
            filtered_accessions[filtered_accessions["accession_category"].str.contains("NEW HIRE", na=False)]["count"].sum()
            / filtered_accessions["count"].sum() * 100
        )
        st.metric("New Hires %", f"{new_hire_pct:.1f}%")

    st.divider()

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            accession_categories(filtered_accessions),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            top_hiring_agencies(filtered_accessions),
            use_container_width=True,
        )

    # Charts row 2
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            education_distribution(filtered_accessions, title="Education Levels of New Hires"),
            use_container_width=True,
        )

    with col2:
        # Geographic distribution of accessions
        acc_by_state = (
            filtered_accessions[filtered_accessions["duty_station_state"] != "REDACTED"]
            .groupby("duty_station_state")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        import plotly.express as px
        fig = px.bar(
            acc_by_state,
            x="count",
            y="duty_station_state",
            orientation="h",
            title="Top 15 States for New Hires",
            labels={"count": "New Hires", "duty_station_state": "State"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TURNOVER TAB
# =============================================================================
with tab_turnover:
    st.header("Turnover Analysis (Separations)")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Total Separations", filtered_separations["count"].sum())
    with col2:
        avg_salary_sep = filtered_separations["annualized_adjusted_basic_pay"].mean()
        st.metric("Avg Salary (Departures)", f"${avg_salary_sep:,.0f}" if avg_salary_sep else "N/A")
    with col3:
        avg_tenure = filtered_separations["length_of_service_years"].mean()
        st.metric("Avg Tenure at Exit", f"{avg_tenure:.1f} years" if avg_tenure else "N/A")
    with col4:
        retirement_pct = (
            filtered_separations[filtered_separations["separation_category"].str.contains("RETIREMENT", na=False)]["count"].sum()
            / filtered_separations["count"].sum() * 100
        )
        st.metric("Retirement %", f"{retirement_pct:.1f}%")

    st.divider()

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            separation_categories(filtered_separations),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            turnover_by_agency(filtered_accessions, filtered_separations),
            use_container_width=True,
        )

    # Charts row 2
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            tenure_at_separation(filtered_separations),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            retirement_analysis(filtered_separations),
            use_container_width=True,
        )

    # Net change chart
    st.plotly_chart(
        net_change_by_agency(filtered_accessions, filtered_separations),
        use_container_width=True,
    )

# =============================================================================
# AI ANALYST TAB
# =============================================================================
with tab_ai:
    st.header("AI Analyst")
    st.caption("Ask questions about the federal workforce data in natural language")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""

    if not api_key:
        st.warning(
            "OpenAI API key not found. Please set `OPENAI_API_KEY` in your environment "
            "or add it to `.streamlit/secrets.toml`."
        )
        st.code("OPENAI_API_KEY = 'sk-...'", language="toml")
    else:
        # Example queries
        st.subheader("Example Questions")
        example_cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUERIES[:6]):
            col = example_cols[i % 2]
            if col.button(example, key=f"example_{i}"):
                st.session_state["ai_question"] = example

        st.divider()

        # Question input
        question = st.text_input(
            "Ask a question",
            value=st.session_state.get("ai_question", ""),
            placeholder="e.g., What is the average salary by grade?",
            key="ai_input",
        )

        if st.button("Analyze", type="primary", disabled=not question):
            with st.spinner("Analyzing..."):
                try:
                    result = query_ai(
                        question=question,
                        employment=filtered_employment,
                        accessions=filtered_accessions,
                        separations=filtered_separations,
                    )

                    # Show explanation
                    st.subheader("Answer")
                    st.write(result.explanation)

                    # Show chart if available
                    if result.execution.figure:
                        st.plotly_chart(result.execution.figure, use_container_width=True)

                    # Show data if available
                    if result.execution.dataframe is not None:
                        st.subheader("Data")
                        st.dataframe(result.execution.dataframe.head(20), use_container_width=True)
                    elif result.execution.result is not None:
                        st.subheader("Result")
                        st.write(result.execution.result)

                    # Show code in expander
                    with st.expander("View generated code"):
                        st.code(result.code, language="python")

                    # Show errors if any
                    if not result.execution.success:
                        st.error(f"Execution error: {result.execution.error}")

                except Exception as e:
                    st.error(f"Error: {e}")

        # Clear question button
        if st.session_state.get("ai_question"):
            if st.button("Clear"):
                st.session_state["ai_question"] = ""
                st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.caption(
    "Data source: [OPM FedScope](https://www.opm.gov/data/Index.aspx) | "
    "Built with Streamlit"
)
