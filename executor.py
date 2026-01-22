"""Safe execution environment for AI-generated pandas code."""
import ast
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Dangerous names to block
BLOCKED_NAMES = {
    "exec", "eval", "compile", "open", "input", "__import__",
    "globals", "locals", "getattr", "setattr", "delattr",
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "urllib", "requests", "pickle", "importlib",
}

BLOCKED_ATTRIBUTES = {
    "__class__", "__bases__", "__subclasses__", "__code__",
    "__globals__", "__builtins__", "__dict__",
}


class CodeValidator(ast.NodeVisitor):
    """AST visitor to validate generated code for safety."""

    def __init__(self):
        self.errors = []

    def visit_Name(self, node):
        if node.id in BLOCKED_NAMES:
            self.errors.append(f"Blocked: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in BLOCKED_ATTRIBUTES or node.attr.startswith("__"):
            self.errors.append(f"Blocked attribute: {node.attr}")
        self.generic_visit(node)


def validate_code(code: str) -> list[str]:
    """Validate code using AST analysis."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]
    validator = CodeValidator()
    validator.visit(tree)
    return validator.errors


@dataclass
class ExecutionResult:
    success: bool
    result: Any = None
    figure: go.Figure | None = None
    dataframe: pd.DataFrame | None = None
    error: str | None = None


def execute_code(code: str, dataframes: dict[str, pd.DataFrame]) -> ExecutionResult:
    """Safely execute generated pandas code."""
    errors = validate_code(code)
    if errors:
        return ExecutionResult(success=False, error="Validation failed: " + "; ".join(errors))

    namespace = {"pd": pd, "np": np, "px": px, "go": go, **dataframes}

    try:
        exec(code, namespace)
    except Exception as e:
        return ExecutionResult(success=False, error=f"{type(e).__name__}: {e}")

    # Extract results (use explicit None checks to avoid Series truthiness issues)
    figure = namespace.get("fig") or namespace.get("figure")
    result = namespace.get("result") or namespace.get("answer")

    result_df = None
    if isinstance(result, pd.DataFrame):
        result_df = result
        result = None
    elif isinstance(result, pd.Series):
        result_df = result.reset_index()
        result = None

    return ExecutionResult(success=True, result=result, figure=figure, dataframe=result_df)


def clean_code_block(code: str) -> str:
    """Clean markdown code blocks from LLM response."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()
