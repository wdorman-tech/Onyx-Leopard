from __future__ import annotations

import json

try:
    from camel.toolkits import FunctionTool

    CAMEL_TOOLS_AVAILABLE = True
except ImportError:
    CAMEL_TOOLS_AVAILABLE = False


def _validate_dept_index(dept_index: int) -> str | None:
    """Return an error JSON string if dept_index is out of range, else None."""
    if not (0 <= dept_index <= 11):
        return json.dumps({"error": "dept_index must be between 0 and 11"})
    return None


def adjust_department_budget(dept_index: int, delta_pct: float) -> str:
    """Adjust a department's budget by percentage. delta_pct: -50 to +100."""
    if not (-50 <= delta_pct <= 100):
        return json.dumps({"error": "delta_pct must be between -50 and +100"})
    if err := _validate_dept_index(dept_index):
        return err
    return json.dumps({
        "tool": "adjust_department_budget",
        "dept_index": dept_index,
        "delta_pct": delta_pct,
        "status": "validated",
    })


def hire_employees(dept_index: int, count: int) -> str:
    """Hire employees for a department. Count: 1-20."""
    if not (1 <= count <= 20):
        return json.dumps({"error": "count must be between 1 and 20"})
    if err := _validate_dept_index(dept_index):
        return err
    return json.dumps({
        "tool": "hire_employees",
        "dept_index": dept_index,
        "count": count,
        "status": "validated",
    })


def fire_employees(dept_index: int, count: int) -> str:
    """Reduce headcount for a department. Count: 1-20."""
    if not (1 <= count <= 20):
        return json.dumps({"error": "count must be between 1 and 20"})
    if err := _validate_dept_index(dept_index):
        return err
    return json.dumps({
        "tool": "fire_employees",
        "dept_index": dept_index,
        "count": count,
        "status": "validated",
    })


def set_price_adjustment(delta_pct: float) -> str:
    """Adjust product pricing by percentage. -10 to +10."""
    if not (-10 <= delta_pct <= 10):
        return json.dumps({"error": "delta_pct must be between -10 and +10"})
    return json.dumps({
        "tool": "set_price_adjustment",
        "delta_pct": delta_pct,
        "status": "validated",
    })


def issue_directive(directive: str) -> str:
    """Executive-only: issue a directive constraining other departments."""
    if not directive or not directive.strip():
        return json.dumps({"error": "directive must be a non-empty string"})
    return json.dumps({
        "tool": "issue_directive",
        "directive": directive.strip(),
        "status": "validated",
    })


def invest_in_capacity(amount: float) -> str:
    """Invest capital to increase production capacity."""
    if amount <= 0:
        return json.dumps({"error": "amount must be positive"})
    return json.dumps({
        "tool": "invest_in_capacity",
        "amount": amount,
        "status": "validated",
    })


def build_simulation_tools() -> list:
    """Build CAMEL FunctionTool list. Returns empty list if camel not available."""
    if not CAMEL_TOOLS_AVAILABLE:
        return []
    return [
        FunctionTool(adjust_department_budget),
        FunctionTool(hire_employees),
        FunctionTool(fire_employees),
        FunctionTool(set_price_adjustment),
        FunctionTool(issue_directive),
        FunctionTool(invest_in_capacity),
    ]
