"""AI query pipeline for workforce analytics using pandas code generation."""
from dataclasses import dataclass
import pandas as pd
from openai import OpenAI

from data_loader import get_schema_for_ai, get_sample_values
from executor import execute_code, clean_code_block, ExecutionResult

SYSTEM_PROMPT = """You are a data analyst. Write Python code using pandas and plotly to answer questions.

Available DataFrames:
- `employment`: Federal workforce (2M+ employees)
- `accessions`: New hires (13K records)
- `separations`: Departures (21K records)

Rules:
- Store results in `result` variable
- Store charts in `fig` variable
- Use plotly express (px) for visualizations
- Sum the `count` column for totals, don't count rows
- Filter out NaN values before analysis
- Output raw Python code only, no markdown
"""


@dataclass
class QueryResult:
    question: str
    explanation: str
    code: str
    execution: ExecutionResult


def query_ai(question: str, employment: pd.DataFrame, accessions: pd.DataFrame,
             separations: pd.DataFrame, model: str = "gpt-4o-mini") -> QueryResult:
    """Query the AI to analyze workforce data."""
    client = OpenAI()

    # Build context
    schemas = "\n".join([
        get_schema_for_ai(employment, "employment"),
        get_schema_for_ai(accessions, "accessions"),
        get_schema_for_ai(separations, "separations"),
    ])

    sample_cols = ["agency", "separation_category", "accession_category", "education_level", "age_bracket"]
    samples = []
    for col in sample_cols:
        for df in [employment, accessions, separations]:
            if col in df.columns:
                samples.append(f"- {col}: {get_sample_values(df, col, 3)}")
                break

    prompt = f"{schemas}\n\nSample values:\n{chr(10).join(samples)}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    code = clean_code_block(response.choices[0].message.content)
    execution = execute_code(code, {"employment": employment, "accessions": accessions, "separations": separations})

    # Generate explanation
    if execution.success:
        result_str = str(execution.dataframe.head(5)) if execution.dataframe is not None else str(execution.result)
        expl_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Briefly explain this result for '{question}':\n{result_str}"}],
            temperature=0,
            max_tokens=200,
        )
        explanation = expl_response.choices[0].message.content
    else:
        explanation = f"Error: {execution.error}"

    return QueryResult(question=question, explanation=explanation, code=code, execution=execution)


EXAMPLE_QUERIES = [
    "What is the average salary by grade?",
    "Which agencies have the highest turnover rate?",
    "What percentage of separations are retirements?",
    "Compare education levels between new hires and those leaving",
    "Show the age distribution of the federal workforce",
    "Which states have the most federal employees?",
]
