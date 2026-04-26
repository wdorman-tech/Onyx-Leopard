"""Bridge between company node graphs and market competition variables.

Auto-derives every modifier set per tick from the actual spawned nodes'
`modifier_keys` (declared in `node_library.yaml`). No static declaration;
by construction this eliminates the entire empty-modifier-key class of
silent-failure bugs (V2 plan §6).

──────────────────────────────────────────────────────────────────────────
Suffix taxonomy (v2) — PART OF THIS FILE'S CONTRACT
──────────────────────────────────────────────────────────────────────────

`bucket_modifiers` partitions an aggregated `dict[str, float]` of modifier
keys into seven buckets by suffix match. A key K matches bucket tag T iff:

    K == T          (exact match — the whole modifier_key IS the tag)
    K endswith "_T" (snake_case suffix — T is the trailing token chain)

The first matching tag (in the order listed below) wins. Anything that
matches no tag goes to `other`. Phase 1.4 (P-ENG-6) introduced `cost`,
`revenue`, and `capital` so the production node library no longer leaves
quantities in `other` that the engine never reads.

    quality       : "quality", "satisfaction", "retention", "churn_reduction"
    marketing     : "marketing", "lead_gen", "pipeline_strength",
                    "brand", "awareness"
    infrastructure: "infrastructure", "capacity_uplift", "throughput",
                    "efficiency"
    cost          : "cost", "cost_reduction", "savings", "overhead", "waste"
    revenue       : "revenue", "monetization", "upsell", "expansion",
                    "premium", "revenue_boost"
    capital       : "capital", "runway", "runway_extension", "funding"
    other         : everything else

Sign conventions consumed by `unified_v2.py`:
  * `cost` bucket: signed sum. Negative total = savings (lowers daily_burn);
    positive total = added cost. Authors often encode reductions as
    negative magnitudes ("cloud_cost_reduction": -0.12).
  * `revenue` bucket: signed sum. Positive total = revenue uplift fraction.
  * `capital` bucket: magnitude (abs of signed sum) = daily access-to-
    capital signal, gated on stance archetype (only venture_growth /
    consolidator can convert it to cash).

Author convention for `node_library.yaml`: pick a modifier_key name whose
trailing token (or the whole name, for short names) matches the bucket the
modifier should land in. Anything unprincipled lands in `other` and is then
the caller's problem to interpret.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.simulation.library_loader import NodeLibrary

# ─────────────────────────────────────────────────────────────────────────────
# v2 — auto-derived modifier aggregation
# ─────────────────────────────────────────────────────────────────────────────

#: Bucket tag tables. Order of buckets in `_BUCKETS` is the priority order:
#: a key that could match more than one bucket lands in the first match.
#: Tags within a bucket are tried longest-first to avoid `_efficiency`
#: shadowing a hypothetical `_throughput_efficiency` (none today, but the
#: rule is stable as the library grows).
QUALITY_TAGS: tuple[str, ...] = (
    "churn_reduction",  # 2-token, must precede single-token "retention" etc.
    "satisfaction",
    "retention",
    "quality",
)
MARKETING_TAGS: tuple[str, ...] = (
    "pipeline_strength",  # 2-token first
    "lead_gen",
    "awareness",
    "marketing",
    "brand",
)
INFRASTRUCTURE_TAGS: tuple[str, ...] = (
    "capacity_uplift",  # 2-token first
    "infrastructure",
    "throughput",
    "efficiency",
)
COST_TAGS: tuple[str, ...] = (
    "cost_reduction",  # 2-token first — beats "cost" alone
    "overhead",
    "savings",
    "waste",
    "cost",
)
REVENUE_TAGS: tuple[str, ...] = (
    "revenue_boost",  # 2-token first
    "monetization",
    "expansion",
    "upsell",
    "premium",
    "revenue",
)
CAPITAL_TAGS: tuple[str, ...] = (
    "runway_extension",  # 2-token first
    "runway",
    "funding",
    "capital",
)

#: Bucket priority order — first match wins. Quality precedes infrastructure
#: (so a hypothetical `*_quality_efficiency` lands in quality, not infra).
#: Cost/revenue/capital sit after the "modifier" buckets so a key like
#: `marketing_cost` (not currently in the library) would still go to the
#: marketing modifier bucket if added later.
_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("quality", QUALITY_TAGS),
    ("marketing", MARKETING_TAGS),
    ("infrastructure", INFRASTRUCTURE_TAGS),
    ("revenue", REVENUE_TAGS),
    ("capital", CAPITAL_TAGS),
    ("cost", COST_TAGS),
)


class BridgeAggregate(BaseModel):
    """Bucketed view of a company's auto-derived modifier set.

    Produced by `bucket_modifiers` from the dict that `aggregate_modifiers`
    returns. All four buckets are `dict[str, float]`; empty dicts are
    expected (early ticks where only the founder is spawned will populate
    nothing).
    """

    model_config = ConfigDict(extra="forbid")

    quality: dict[str, float] = Field(default_factory=dict)
    marketing: dict[str, float] = Field(default_factory=dict)
    infrastructure: dict[str, float] = Field(default_factory=dict)
    cost: dict[str, float] = Field(default_factory=dict)
    revenue: dict[str, float] = Field(default_factory=dict)
    capital: dict[str, float] = Field(default_factory=dict)
    other: dict[str, float] = Field(default_factory=dict)


def aggregate_modifiers(
    library: NodeLibrary,
    spawned_nodes: dict[str, int],
) -> dict[str, float]:
    """Sum each modifier_key across all spawned nodes (count-weighted).

    For each `(node_key, count)` in `spawned_nodes`, looks up
    `library.nodes[node_key].modifier_keys`, multiplies each value by `count`,
    and sums into a running dict.

    Args:
        library: Validated node library (typically from `get_library()`).
        spawned_nodes: Map of `node_key -> count` of currently-spawned nodes.
            Counts of 0 are skipped; negative counts raise `ValueError`.

    Returns:
        Mapping of `modifier_key -> summed magnitude`. Empty dict if
        `spawned_nodes` is empty or no spawned node carries any modifiers.

    Raises:
        KeyError: if any `node_key` in `spawned_nodes` is not present in
            `library` — V2 plan §1 hard-fail validation rule.
        ValueError: if any count is negative.
    """
    out: dict[str, float] = {}
    for node_key, count in spawned_nodes.items():
        if count < 0:
            raise ValueError(
                f"spawned_nodes[{node_key!r}] = {count}; counts cannot be negative"
            )
        if count == 0:
            continue
        node = library.get(node_key)
        if node is None:
            raise KeyError(
                f"spawned_nodes references unknown node_key {node_key!r} "
                "(not present in node library). V2 plan: hard-fail validation."
            )
        for mod_key, magnitude in node.modifier_keys.items():
            out[mod_key] = out.get(mod_key, 0.0) + magnitude * count
    return out


def _classify_modifier_key(key: str) -> str:
    """Return bucket name (`quality` / `marketing` / `infrastructure` / `other`).

    Strict semantics: a tag T matches `key` iff `key == T` or `key.endswith("_" + T)`.
    First bucket (in priority order) with any matching tag wins. Within a
    bucket, tags are searched longest-first so multi-word tags (e.g.
    `pipeline_strength`) win over their single-word substrings.
    """
    for bucket_name, tags in _BUCKETS:
        for tag in tags:
            if key == tag or key.endswith("_" + tag):
                return bucket_name
    return "other"


def bucket_modifiers(modifiers: dict[str, float]) -> BridgeAggregate:
    """Partition an aggregated modifier dict into four buckets by suffix.

    See module docstring for the suffix taxonomy. The taxonomy is the file's
    contract — `node_library.yaml` authors should pick modifier_key names
    that match the bucket they're meant for.
    """
    buckets: dict[str, dict[str, float]] = {
        "quality": {},
        "marketing": {},
        "infrastructure": {},
        "cost": {},
        "revenue": {},
        "capital": {},
        "other": {},
    }
    for key, val in modifiers.items():
        buckets[_classify_modifier_key(key)][key] = val
    return BridgeAggregate(**buckets)


def derive_bridge_aggregate(
    library: NodeLibrary,
    spawned_nodes: dict[str, int],
) -> BridgeAggregate:
    """One-shot helper: aggregate then bucket. The hot path for v2 callers."""
    return bucket_modifiers(aggregate_modifiers(library, spawned_nodes))



__all__ = [
    "CAPITAL_TAGS",
    "COST_TAGS",
    "INFRASTRUCTURE_TAGS",
    "MARKETING_TAGS",
    "QUALITY_TAGS",
    "REVENUE_TAGS",
    "BridgeAggregate",
    "aggregate_modifiers",
    "bucket_modifiers",
    "derive_bridge_aggregate",
]
