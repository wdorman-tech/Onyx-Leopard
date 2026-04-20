from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _safe_get(snapshot: dict, key: str, company_idx: int, default: Any = 0.0) -> Any:
    """Extract a per-company value from snapshot, handling both list and scalar formats."""
    val = snapshot.get(key)
    if val is None:
        return default
    if isinstance(val, list):
        indices = snapshot.get("indices", [])
        if company_idx in indices:
            pos = indices.index(company_idx)
            return val[pos] if pos < len(val) else default
        return default
    return val



def heuristic_finance(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Budget reallocation based on revenue-to-cost ratio."""
    revenue = _safe_get(snapshot, "revenue", company_idx, 1.0)
    costs = _safe_get(snapshot, "costs", company_idx, 1.0)
    ratio = revenue / max(costs, 1.0)

    if ratio > 1.2:
        return {
            "action": "adjust_budget",
            "parameters": {"dept": 1, "delta_pct": 5.0},
            "rationale": "Revenue growing, increase R&D",
            "confidence": 0.7,
        }
    if ratio < 0.8:
        return {
            "action": "adjust_budget",
            "parameters": {"dept": -1, "delta_pct": -5.0},
            "rationale": "Costs exceeding revenue, cut lowest-performing dept",
            "confidence": 0.65,
        }
    return {
        "action": "hold",
        "parameters": {},
        "rationale": "Revenue/cost ratio stable",
        "confidence": 0.8,
    }


def heuristic_rd(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Innovation direction based on market position."""
    market_share = _safe_get(snapshot, "market_share", company_idx, 0.1)
    health = _safe_get(snapshot, "health_score", company_idx, 0.5)

    if market_share < 0.1:
        return {
            "action": "pivot_innovation",
            "parameters": {"focus": "disruptive", "intensity": 0.8},
            "rationale": "Low market share, pursue disruptive innovation",
            "confidence": 0.6,
        }
    if health > 0.7:
        return {
            "action": "sustain_innovation",
            "parameters": {"focus": "incremental", "intensity": 0.5},
            "rationale": "Healthy position, incremental improvements",
            "confidence": 0.75,
        }
    return {
        "action": "hold",
        "parameters": {"focus": "maintenance", "intensity": 0.3},
        "rationale": "Mid-range position, maintain current R&D",
        "confidence": 0.7,
    }


def heuristic_distribution(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Route optimization placeholder."""
    firm_size = _safe_get(snapshot, "firm_size", company_idx, 10.0)

    if firm_size > 30.0:
        return {
            "action": "expand_routes",
            "parameters": {"new_routes": 2},
            "rationale": "Large firm, expand distribution network",
            "confidence": 0.6,
        }
    return {
        "action": "hold",
        "parameters": {},
        "rationale": "Current distribution adequate for firm size",
        "confidence": 0.75,
    }


def heuristic_production(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Capacity planning based on utilization."""
    labor = _safe_get(snapshot, "labor", company_idx, 50.0)
    capital = _safe_get(snapshot, "capital", company_idx, 1e6)
    utilization = labor * 1000 / max(capital, 1.0)

    if utilization > 0.85:
        return {
            "action": "expand_capacity",
            "parameters": {"capital_increase_pct": 10.0},
            "rationale": "High utilization, expand production capacity",
            "confidence": 0.7,
        }
    if utilization < 0.5:
        return {
            "action": "reduce_capacity",
            "parameters": {"capital_decrease_pct": 5.0},
            "rationale": "Low utilization, reduce excess capacity",
            "confidence": 0.65,
        }
    return {
        "action": "hold",
        "parameters": {},
        "rationale": "Production utilization within normal range",
        "confidence": 0.8,
    }


def heuristic_sales(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Price adjustment within +/-5% based on market share."""
    market_share = _safe_get(snapshot, "market_share", company_idx, 0.1)

    if market_share > 0.25:
        return {
            "action": "adjust_price",
            "parameters": {"delta_pct": 3.0},
            "rationale": "Strong market position, increase prices",
            "confidence": 0.7,
        }
    if market_share < 0.05:
        return {
            "action": "adjust_price",
            "parameters": {"delta_pct": -5.0},
            "rationale": "Low market share, discount to gain customers",
            "confidence": 0.65,
        }
    return {
        "action": "adjust_price",
        "parameters": {"delta_pct": 0.0},
        "rationale": "Market share in acceptable range, hold prices",
        "confidence": 0.75,
    }


def heuristic_marketing(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Campaign budget based on growth stage."""
    firm_size = _safe_get(snapshot, "firm_size", company_idx, 10.0)
    growth_rate = _safe_get(snapshot, "growth_rate", company_idx, 0.05)

    if growth_rate > 0.06 and firm_size < 20.0:
        return {
            "action": "increase_campaign",
            "parameters": {"budget_increase_pct": 15.0, "channel": "digital"},
            "rationale": "Growth phase, invest heavily in marketing",
            "confidence": 0.7,
        }
    if firm_size > 30.0:
        return {
            "action": "maintain_campaign",
            "parameters": {"budget_increase_pct": 2.0, "channel": "brand"},
            "rationale": "Mature firm, focus on brand maintenance",
            "confidence": 0.75,
        }
    return {
        "action": "hold",
        "parameters": {"budget_increase_pct": 0.0},
        "rationale": "Standard marketing posture",
        "confidence": 0.7,
    }


def heuristic_hr(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Headcount: hire if revenue/employee > threshold, fire if below."""
    revenue = _safe_get(snapshot, "revenue", company_idx, 1e5)
    labor = _safe_get(snapshot, "labor", company_idx, 50.0)
    rev_per_employee = revenue / max(labor, 1.0)

    if rev_per_employee > 5000.0:
        return {
            "action": "hire",
            "parameters": {"headcount_delta": 5},
            "rationale": "High revenue per employee, hire to support growth",
            "confidence": 0.7,
        }
    if rev_per_employee < 1500.0:
        return {
            "action": "layoff",
            "parameters": {"headcount_delta": -3},
            "rationale": "Low revenue per employee, reduce headcount",
            "confidence": 0.6,
        }
    return {
        "action": "hold",
        "parameters": {"headcount_delta": 0},
        "rationale": "Revenue per employee within normal band",
        "confidence": 0.8,
    }


def heuristic_executive(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Strategic posture based on health score and market position."""
    health = _safe_get(snapshot, "health_score", company_idx, 0.5)
    market_share = _safe_get(snapshot, "market_share", company_idx, 0.1)

    if health > 0.7 and market_share > 0.2:
        return {
            "action": "aggressive_growth",
            "parameters": {"posture": "expand"},
            "rationale": "Strong health and market position, pursue growth",
            "confidence": 0.75,
        }
    if health < 0.3:
        return {
            "action": "defensive",
            "parameters": {"posture": "consolidate"},
            "rationale": "Poor health score, consolidate operations",
            "confidence": 0.7,
        }
    return {
        "action": "steady",
        "parameters": {"posture": "maintain"},
        "rationale": "Moderate position, maintain current strategy",
        "confidence": 0.7,
    }


def heuristic_customer_service(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Service level targets based on headcount ratio."""
    labor = _safe_get(snapshot, "labor", company_idx, 50.0)
    firm_size = _safe_get(snapshot, "firm_size", company_idx, 10.0)
    service_ratio = labor / max(firm_size, 1.0)

    if service_ratio < 2.0:
        return {
            "action": "increase_service_staff",
            "parameters": {"hire_count": 3},
            "rationale": "Low service staff ratio, risk of poor satisfaction",
            "confidence": 0.65,
        }
    if service_ratio > 8.0:
        return {
            "action": "reduce_service_staff",
            "parameters": {"reduce_count": 2},
            "rationale": "Overstaffed service department",
            "confidence": 0.6,
        }
    return {
        "action": "hold",
        "parameters": {},
        "rationale": "Service staffing adequate",
        "confidence": 0.75,
    }


def heuristic_legal(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Compliance posture -- mostly hold steady."""
    firm_size = _safe_get(snapshot, "firm_size", company_idx, 10.0)

    if firm_size > 35.0:
        return {
            "action": "increase_compliance",
            "parameters": {"audit_frequency": "quarterly"},
            "rationale": "Large firm, increase compliance oversight",
            "confidence": 0.8,
        }
    return {
        "action": "hold",
        "parameters": {"audit_frequency": "annual"},
        "rationale": "Standard compliance posture sufficient",
        "confidence": 0.85,
    }


def heuristic_it(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Infrastructure investment based on firm size."""
    firm_size = _safe_get(snapshot, "firm_size", company_idx, 10.0)
    labor = _safe_get(snapshot, "labor", company_idx, 50.0)

    if firm_size > 25.0 and labor > 100.0:
        return {
            "action": "upgrade_infrastructure",
            "parameters": {"investment_pct": 8.0},
            "rationale": "Growing firm needs infrastructure investment",
            "confidence": 0.7,
        }
    return {
        "action": "hold",
        "parameters": {"investment_pct": 2.0},
        "rationale": "Minimal IT investment sufficient",
        "confidence": 0.75,
    }


def heuristic_procurement(snapshot: dict, company_idx: int) -> dict[str, Any]:
    """Vendor cost optimization."""
    costs = _safe_get(snapshot, "costs", company_idx, 1e5)
    revenue = _safe_get(snapshot, "revenue", company_idx, 1e5)
    cost_ratio = costs / max(revenue, 1.0)

    if cost_ratio > 0.7:
        return {
            "action": "renegotiate_vendors",
            "parameters": {"target_savings_pct": 5.0},
            "rationale": "High cost ratio, renegotiate vendor contracts",
            "confidence": 0.65,
        }
    return {
        "action": "hold",
        "parameters": {},
        "rationale": "Procurement costs within acceptable range",
        "confidence": 0.8,
    }


HEURISTIC_REGISTRY: dict[int, Callable[[dict, int], dict[str, Any]]] = {
    0: heuristic_finance,
    1: heuristic_rd,
    2: heuristic_distribution,
    3: heuristic_production,
    4: heuristic_sales,
    5: heuristic_marketing,
    6: heuristic_hr,
    7: heuristic_executive,
    8: heuristic_customer_service,
    9: heuristic_legal,
    10: heuristic_it,
    11: heuristic_procurement,
}
