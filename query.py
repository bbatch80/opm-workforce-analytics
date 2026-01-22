"""AI query pipeline for workforce analytics using pandas code generation."""
import os
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI

from data_loader import get_schema_for_ai, get_sample_values
from executor import execute_code, clean_code_block, ExecutionResult


SYSTEM_PROMPT = """You are a data analyst assistant that helps users analyze federal workforce data.
You write Python code using pandas, numpy, and plotly to answer questions.

## Available DataFrames

You have access to these pre-loaded DataFrames:
- `employment`: Current federal workforce snapshot (2M+ employees)
- `accessions`: New hires entering federal service (13,838 records)
- `separations`: Employees leaving federal service (20,976 records)

## Important Notes

1. **Numeric columns**: `annualized_adjusted_basic_pay` and `length_of_service_years` are already numeric (may have NaN for REDACTED values)
2. **Count column**: Each row has a `count` column (usually 1). Sum this column for totals, don't count rows.
3. **Output requirements**:
   - Store the main result in a variable called `result` (can be a DataFrame, number, or string)
   - If creating a chart, store the plotly figure in a variable called `fig`
   - Use plotly express (px) or plotly graph objects (go) for visualizations

## Code Guidelines

- Always filter out NaN and REDACTED values before analysis
- Use descriptive variable names
- Keep code concise but readable
- For "top N" questions, use `.nlargest()` or `.nsmallest()`
- For percentages, calculate from count sums, not row counts

## Example Patterns

**Average salary by grade:**
```python
result = employment[employment['annualized_adjusted_basic_pay'].notna()].groupby('grade')['annualized_adjusted_basic_pay'].mean().sort_values(ascending=False)
```

**Top agencies by turnover (separations/workforce ratio):**
```python
emp_by_agency = employment.groupby('agency')['count'].sum()
sep_by_agency = separations.groupby('agency')['count'].sum()
turnover = (sep_by_agency / emp_by_agency * 100).dropna().sort_values(ascending=False)
result = turnover.head(10)
```

**Create a bar chart:**
```python
import plotly.express as px
data = employment.groupby('agency')['count'].sum().nlargest(10).reset_index()
fig = px.bar(data, x='agency', y='count', title='Top 10 Agencies')
result = data
```
"""


@dataclass
class QueryResult:
    """Result of an AI query."""
    question: str
    explanation: str
    code: str
    execution: ExecutionResult


def build_prompt(question: str, employment: pd.DataFrame, accessions: pd.DataFrame, separations: pd.DataFrame) -> str:
    """Build the full prompt with schema context.

    Args:
        question: User's natural language question
        employment: Employment DataFrame
        accessions: Accessions DataFrame
        separations: Separations DataFrame

    Returns:
        Complete prompt string
    """
    # Get schema info
    schemas = [
        get_schema_for_ai(employment, "employment"),
        get_schema_for_ai(accessions, "accessions"),
        get_schema_for_ai(separations, "separations"),
    ]

    # Get sample values for key categorical columns
    sample_info = []
    for col in ["agency", "separation_category", "accession_category", "education_level", "age_bracket"]:
        if col in employment.columns:
            values = get_sample_values(employment, col, 5)
            sample_info.append(f"- {col}: {values}")
        elif col in accessions.columns:
            values = get_sample_values(accessions, col, 5)
            sample_info.append(f"- {col}: {values}")
        elif col in separations.columns:
            values = get_sample_values(separations, col, 5)
            sample_info.append(f"- {col}: {values}")

    prompt = f"""## Data Schemas

{chr(10).join(schemas)}

## Sample Values for Key Columns
{chr(10).join(sample_info)}

## User Question
{question}

## Instructions
Write Python code to answer this question. Use pandas, numpy, and plotly.
- Store results in `result` variable
- Store any chart in `fig` variable
- DO NOT include markdown code blocks in your response
- Just write the raw Python code
"""
    return prompt


def query_ai(
    question: str,
    employment: pd.DataFrame,
    accessions: pd.DataFrame,
    separations: pd.DataFrame,
    model: str = "gpt-4o-mini",
) -> QueryResult:
    """Query the AI to analyze workforce data.

    Args:
        question: Natural language question
        employment: Employment DataFrame
        accessions: Accessions DataFrame
        separations: Separations DataFrame
        model: OpenAI model to use

    Returns:
        QueryResult with explanation, code, and execution result
    """
    client = OpenAI()

    # Build prompt with context
    user_prompt = build_prompt(question, employment, accessions, separations)

    # Call OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=2000,
    )

    # Extract code from response
    raw_response = response.choices[0].message.content
    code = clean_code_block(raw_response)

    # Execute the code
    dataframes = {
        "employment": employment,
        "accessions": accessions,
        "separations": separations,
    }
    execution = execute_code(code, dataframes)

    # Generate explanation
    explanation = generate_explanation(question, code, execution, client, model)

    return QueryResult(
        question=question,
        explanation=explanation,
        code=code,
        execution=execution,
    )


def generate_explanation(
    question: str,
    code: str,
    execution: ExecutionResult,
    client: OpenAI,
    model: str,
) -> str:
    """Generate a natural language explanation of the results.

    Args:
        question: Original question
        code: Generated code
        execution: Execution result
        client: OpenAI client
        model: Model to use

    Returns:
        Natural language explanation
    """
    if not execution.success:
        return f"The analysis encountered an error: {execution.error}"

    # Build result summary
    result_summary = ""
    if execution.dataframe is not None:
        result_summary = f"DataFrame result:\n{execution.dataframe.head(10).to_string()}"
    elif execution.result is not None:
        result_summary = f"Result: {execution.result}"

    explanation_prompt = f"""The user asked: "{question}"

The analysis code produced this result:
{result_summary}

Write a brief (2-3 sentences) explanation of what this data shows. Be specific about the numbers.
Focus on the key insight that answers the user's question."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You explain data analysis results clearly and concisely."},
            {"role": "user", "content": explanation_prompt},
        ],
        temperature=0,
        max_tokens=300,
    )

    return response.choices[0].message.content


# Example queries for the UI
EXAMPLE_QUERIES = [
    "What is the average salary by grade?",
    "Which agencies have the highest turnover rate?",
    "What percentage of separations are retirements?",
    "Compare education levels between new hires and those leaving",
    "Show the age distribution of the federal workforce",
    "Which states have the most federal employees?",
    "What are the top reasons people leave federal service?",
    "How does salary vary by education level?",
    "Which agencies are hiring the most?",
    "What is the average length of service for people who quit vs retire?",
]
