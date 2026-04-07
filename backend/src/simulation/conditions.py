"""Data-driven trigger condition evaluator.

Replaces the 19 hardcoded trigger classes with a single function that evaluates
condition dicts from YAML config against computed metrics.
"""

from __future__ import annotations


def evaluate_condition(
    condition: dict,
    metrics: dict[str, float],
    node_type_counts: dict[str, int],
) -> bool:
    """Evaluate a trigger condition dict against current company metrics.

    Supports these patterns:
      - Metric comparison: {"location_count": {">=": 5}}
      - AND combinator: {"all": [cond1, cond2, ...]}
      - OR combinator: {"any": [cond1, cond2, ...]}
      - Node existence: {"has_node": "general_manager"}
      - Node count ratio: {"node_count_ratio": {"node_type": "area_manager", "per": 6}}
    """
    if "all" in condition:
        return all(
            evaluate_condition(c, metrics, node_type_counts)
            for c in condition["all"]
        )

    if "any" in condition:
        return any(
            evaluate_condition(c, metrics, node_type_counts)
            for c in condition["any"]
        )

    if "has_node" in condition:
        return node_type_counts.get(condition["has_node"], 0) > 0

    if "node_count_ratio" in condition:
        spec = condition["node_count_ratio"]
        nt = spec["node_type"]
        per = spec["per"]
        locs = metrics.get("location_count", 0)
        count = node_type_counts.get(nt, 0)
        return locs >= per and locs > count * per

    # Simple metric comparisons: {"metric_name": {">": value}}
    for key, comp in condition.items():
        if not isinstance(comp, dict):
            continue
        val = metrics.get(key, 0.0)
        for op, threshold in comp.items():
            if op == ">" and not (val > threshold):
                return False
            if op == ">=" and not (val >= threshold):
                return False
            if op == "<" and not (val < threshold):
                return False
            if op == "<=" and not (val <= threshold):
                return False
            if op == "==" and val != threshold:
                return False

    return True
