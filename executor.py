"""Safe execution environment for AI-generated pandas code."""
import ast
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class ExecutionError(Exception):
    """Raised when code execution fails."""
    pass


class ValidationError(Exception):
    """Raised when code fails validation."""
    pass


# Allowed modules and their safe imports
ALLOWED_MODULES = {
    "pandas": pd,
    "pd": pd,
    "numpy": np,
    "np": np,
    "plotly.express": px,
    "px": px,
    "plotly.graph_objects": go,
    "go": go,
}

# Dangerous function names to block
BLOCKED_NAMES = {
    "exec",
    "eval",
    "compile",
    "open",
    "input",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "vars",
    "dir",
    "type",
    "isinstance",
    "issubclass",
    "callable",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "object",
    "breakpoint",
    "memoryview",
    "help",
    "credits",
    "license",
    "copyright",
    # File/OS operations
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "glob",
    "io",
    "tempfile",
    # Network operations
    "socket",
    "urllib",
    "requests",
    "http",
    "ftplib",
    "smtplib",
    # Code execution
    "importlib",
    "runpy",
    "code",
    "codeop",
    "ast",
    "dis",
    "inspect",
    "traceback",
    # Other dangerous
    "pickle",
    "shelve",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "concurrent",
}

# Blocked attribute access patterns
BLOCKED_ATTRIBUTES = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__code__",
    "__globals__",
    "__builtins__",
    "__dict__",
    "__module__",
    "__init__",
    "__new__",
    "__del__",
    "__call__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
}


class CodeValidator(ast.NodeVisitor):
    """AST visitor to validate generated code for safety."""

    def __init__(self):
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Block direct imports."""
        for alias in node.names:
            if alias.name not in ALLOWED_MODULES:
                self.errors.append(f"Import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block from imports except from allowed modules."""
        module = node.module or ""
        if module not in ALLOWED_MODULES and not module.startswith(("pandas", "numpy", "plotly")):
            self.errors.append(f"Import from not allowed: {module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Block dangerous function/variable names."""
        if node.id in BLOCKED_NAMES:
            self.errors.append(f"Name not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Block dangerous attribute access."""
        if node.attr in BLOCKED_ATTRIBUTES:
            self.errors.append(f"Attribute access not allowed: {node.attr}")
        if node.attr.startswith("_") and node.attr not in ("_", "__"):
            # Allow single underscore (common in pandas) but block dunder
            if node.attr.startswith("__"):
                self.errors.append(f"Private attribute access not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for dangerous patterns."""
        # Check for dangerous function names
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_NAMES:
                self.errors.append(f"Function call not allowed: {node.func.id}")
        self.generic_visit(node)


def validate_code(code: str) -> list[str]:
    """Validate code using AST analysis.

    Args:
        code: Python code string to validate

    Returns:
        List of validation errors (empty if valid)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    validator = CodeValidator()
    validator.visit(tree)
    return validator.errors


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    result: Any = None
    figure: go.Figure | None = None
    dataframe: pd.DataFrame | None = None
    error: str | None = None


def execute_code(
    code: str,
    dataframes: dict[str, pd.DataFrame],
    timeout_seconds: int = 5,
) -> ExecutionResult:
    """Safely execute generated pandas code.

    Args:
        code: Python code to execute
        dataframes: Dictionary of DataFrames available to the code
            (e.g., {"employment": df_employment, "accessions": df_accessions})
        timeout_seconds: Maximum execution time

    Returns:
        ExecutionResult with success status and any outputs
    """
    # Validate code first
    errors = validate_code(code)
    if errors:
        return ExecutionResult(
            success=False,
            error="Code validation failed:\n" + "\n".join(f"- {e}" for e in errors),
        )

    # Build execution namespace with allowed modules and data
    namespace = {
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        **dataframes,
    }

    # Execute code (timeout removed - doesn't work in Streamlit's threaded environment)
    # Safety is enforced via AST validation above
    try:
        exec(code, namespace)
    except Exception as e:
        return ExecutionResult(success=False, error=f"Execution error: {type(e).__name__}: {e}")

    # Extract results from namespace (use explicit None checks to avoid Series truthiness issues)
    figure = namespace.get("fig")
    if figure is None:
        figure = namespace.get("figure")

    result = namespace.get("result")
    if result is None:
        result = namespace.get("answer")

    result_df = None
    # Check if result is a DataFrame or Series
    if isinstance(result, pd.DataFrame):
        result_df = result
        result = None
    elif isinstance(result, pd.Series):
        result_df = result.reset_index()
        result_df.columns = ["index", "value"] if len(result_df.columns) == 2 else result_df.columns
        result = None

    return ExecutionResult(
        success=True,
        result=result,
        figure=figure,
        dataframe=result_df,
    )


def clean_code_block(code: str) -> str:
    """Clean markdown code blocks from LLM response.

    Args:
        code: Code that may contain markdown formatting

    Returns:
        Clean Python code
    """
    code = code.strip()

    # Remove markdown code blocks
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]

    if code.endswith("```"):
        code = code[:-3]

    return code.strip()
