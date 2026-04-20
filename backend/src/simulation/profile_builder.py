"""AI-powered business profile builder — generates IndustrySpec from user interview."""

from __future__ import annotations

import json
import logging
import re
import uuid

import yaml
from pydantic import BaseModel

from src.simulation.ceo_agent import _DEFAULT_CEO_MODEL as CEO_MODEL, _get_client
from src.simulation.config_loader import INDUSTRY_DIR, IndustrySpec

log = logging.getLogger(__name__)

# ── Session model ──


class InterviewSession(BaseModel):
    id: str
    transcript: list[dict] = []  # [{role: "assistant"/"user", content: "..."}]
    document_summaries: list[dict] = []  # [{filename: str, summary: str}]
    generated_spec: dict | None = None
    status: str = "interviewing"  # "interviewing" | "generating" | "complete" | "error"
    error: str | None = None


# In-memory sessions (fine for local-run open-source project)
_sessions: dict[str, InterviewSession] = {}


def get_session(session_id: str) -> InterviewSession | None:
    return _sessions.get(session_id)


# ── Interview system prompt ──

_INTERVIEW_SYSTEM = """You are a business analyst conducting an interview to build a simulation profile.
Your goal is to understand the user's business well enough to generate a complete
industry simulation configuration.

You need to gather information about:
1. BUSINESS TYPE: What the company does, what it sells, the industry
2. REVENUE MODEL: How money is made (physical sales, subscriptions, services, etc.)
3. ECONOMICS: Pricing, variable costs per unit, fixed costs, supply chain
4. LOCATIONS/UNITS: What a "location" means for this business (store, office, product line, etc.)
5. ORGANIZATIONAL STRUCTURE: Key roles as the company grows (first hires, department heads, etc.)
6. GROWTH MILESTONES: What triggers growth (revenue thresholds, headcount, location count)
7. EXTERNAL DEPENDENCIES: Suppliers, partners, investors
8. REVENUE STREAMS: Beyond the core product (consulting, licensing, etc.)

RULES:
- Ask ONE focused question at a time
- Be conversational but efficient — aim for 8-12 questions total
- Adapt follow-up questions based on answers
- After gathering enough info, say EXACTLY "INTERVIEW_COMPLETE" on its own line,
  followed by a brief summary of what you learned
- Do NOT generate YAML or technical output — just interview naturally
- For the FIRST message, introduce yourself briefly and ask what the business does"""


def start_session() -> InterviewSession:
    """Create a new interview session and generate the first question."""
    session = InterviewSession(id=str(uuid.uuid4()))
    _sessions[session.id] = session
    return session


async def get_first_question(session: InterviewSession) -> str:
    """Get the opening question from Claude."""
    client = _get_client()
    response = await client.messages.create(
        model=CEO_MODEL,
        max_tokens=300,
        system=_INTERVIEW_SYSTEM,
        messages=[{"role": "user", "content": "Start the interview."}],
    )
    question = response.content[0].text
    session.transcript.append({"role": "assistant", "content": question})
    return question


async def process_answer(session: InterviewSession, answer: str) -> dict:
    """Process a user answer and get the next question or complete the interview.

    Returns: {next_question: str, progress: float, is_complete: bool}
    """
    session.transcript.append({"role": "user", "content": answer})

    # Build messages for Claude from transcript
    messages = []
    for entry in session.transcript:
        messages.append({"role": entry["role"], "content": entry["content"]})

    client = _get_client()
    response = await client.messages.create(
        model=CEO_MODEL,
        max_tokens=400,
        system=_INTERVIEW_SYSTEM,
        messages=messages,
    )
    reply = response.content[0].text
    session.transcript.append({"role": "assistant", "content": reply})

    # Check if interview is complete
    is_complete = "INTERVIEW_COMPLETE" in reply
    if is_complete:
        session.status = "generating"

    # Estimate progress (rough: count Q&A pairs, target ~10)
    qa_count = sum(1 for t in session.transcript if t["role"] == "user")
    progress = min(0.95, qa_count / 10.0)
    if is_complete:
        progress = 1.0

    return {
        "next_question": reply,
        "progress": round(progress, 2),
        "is_complete": is_complete,
    }


# ── YAML generation ──

_REFERENCE_YAML = (INDUSTRY_DIR / "restaurant.yaml").read_text()

_GENERATION_SYSTEM = f"""You are a simulation configuration generator. Given an interview transcript
about a business, generate a complete IndustrySpec YAML configuration that can
power a business simulation.

The YAML must follow this exact structure (reference example below).
Every field shown is required unless noted.

KEY RULES:
- meta.slug must be lowercase with underscores only
- roles.location_type and roles.founder_type MUST exist as keys in the nodes dict
- Every trigger node_type MUST exist as a key in the nodes dict
- roles.supplier_types entries MUST exist as keys in the nodes dict
- Exactly ONE trigger must have is_location_expansion: true
- economics_model must be "physical", "subscription", or "service"
- For subscription businesses: set capacity_decay_rate to 0, set churn_rate > 0
- For service businesses: set capacity_decay_rate low (bench time)
- For physical businesses: set capacity_decay_rate > 0 (spoilage/waste)
- Node categories must be: location, corporate, external, or revenue
- Bridge quality_modifier_keys should reference keys that appear in node revenue_modifiers
- Trigger conditions use this DSL:
  Simple: {{"monthly_revenue": {{">": 15000}}}}
  AND: {{"all": [...]}}
  OR: {{"any": [...]}}
  Has node: {{"has_node": "node_type_key"}}

Output ONLY valid YAML — no markdown fences, no explanation, just the YAML.

REFERENCE EXAMPLE (restaurant industry):
```yaml
{_REFERENCE_YAML}
```"""


async def generate_spec(session: InterviewSession) -> dict:
    """Generate an IndustrySpec from the interview transcript + documents.

    Returns the raw dict (YAML-parsed) on success.
    Raises ValueError on failure after retries.
    """
    # Build the user prompt from transcript + docs
    transcript_text = "\n".join(
        f"{'Q' if t['role'] == 'assistant' else 'A'}: {t['content']}"
        for t in session.transcript
    )

    doc_text = ""
    if session.document_summaries:
        doc_text = "\n\nUPLOADED DOCUMENTS:\n"
        for doc in session.document_summaries:
            doc_text += f"\n--- {doc['filename']} ---\n{doc['summary']}\n"

    user_prompt = f"""Based on this business interview, generate a complete IndustrySpec YAML.

INTERVIEW TRANSCRIPT:
{transcript_text}
{doc_text}

Generate the YAML now. Remember: every trigger node_type and role type must exist in the nodes dict."""

    client = _get_client()
    last_error = ""

    for attempt in range(3):
        try:
            retry_hint = ""
            if attempt > 0 and last_error:
                retry_hint = f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n{last_error}\nFix the issue and regenerate."

            response = await client.messages.create(
                model=CEO_MODEL,
                max_tokens=4000,
                system=_GENERATION_SYSTEM,
                messages=[{"role": "user", "content": user_prompt + retry_hint}],
            )
            raw_yaml = response.content[0].text

            # Strip markdown fences if present
            raw_yaml = re.sub(r"^```(?:yaml)?\s*\n", "", raw_yaml, flags=re.MULTILINE)
            raw_yaml = re.sub(r"\n```\s*$", "", raw_yaml, flags=re.MULTILINE)

            parsed = yaml.safe_load(raw_yaml)
            if not isinstance(parsed, dict):
                raise ValueError("YAML did not parse to a dict")

            # Validate with Pydantic
            spec = IndustrySpec(**parsed)

            # Cross-reference validation (same as config_loader)
            if spec.roles.location_type not in spec.nodes:
                raise ValueError(
                    f"roles.location_type '{spec.roles.location_type}' not in nodes"
                )
            if spec.roles.founder_type not in spec.nodes:
                raise ValueError(
                    f"roles.founder_type '{spec.roles.founder_type}' not in nodes"
                )
            for st in spec.roles.supplier_types:
                if st not in spec.nodes:
                    raise ValueError(f"supplier_type '{st}' not in nodes")
            for trigger in spec.triggers:
                if trigger.node_type not in spec.nodes:
                    raise ValueError(
                        f"trigger node_type '{trigger.node_type}' not in nodes"
                    )

            # Ensure exactly one expansion trigger
            expansion = [t for t in spec.triggers if t.is_location_expansion]
            if len(expansion) != 1:
                raise ValueError(
                    f"Expected exactly 1 expansion trigger, got {len(expansion)}"
                )

            session.generated_spec = parsed
            session.status = "complete"
            return parsed

        except Exception as e:
            last_error = str(e)
            log.warning(
                "Spec generation attempt %d/3 failed: %s", attempt + 1, last_error
            )

    session.status = "error"
    session.error = f"Failed to generate valid spec after 3 attempts: {last_error}"
    raise ValueError(session.error)


# ── Adaptive mode: niche analysis + spec generation ──

_NICHE_ANALYSIS_SYSTEM = """You are a business analyst. Given a company name and a detailed description,
identify the precise business niche, provide a rich summary of the business model,
and classify the economics model.

The user may provide extensive detail about their business — pricing, costs, team,
operations, customers, revenue, growth plans. Use ALL of this information to produce
an accurate niche classification and a summary that preserves key financial
and operational details.

Respond with ONLY valid JSON (no markdown fences, no explanation):
{
  "niche": "Specific niche label (e.g. 'Premium DTC Coffee Subscription', 'B2B SaaS Revenue Analytics')",
  "summary": "3-5 sentence summary preserving key details: revenue model, pricing range, cost structure, target customers, team size, and growth stage. Be specific with numbers when the user provides them.",
  "economics_model": "physical OR subscription OR service"
}

Economics model classification:
- "physical": Sells physical goods. Has inventory, spoilage/waste, supply chain.
- "subscription": Recurring revenue. Has churn, MRR/ARR, customer acquisition cost.
- "service": Sells time/expertise. Has billable hours, utilization, bench time."""


async def analyze_niche(company_name: str, description: str) -> dict:
    """1-shot niche identification from company name and description.

    Returns: {niche: str, summary: str, economics_model: str}
    """
    client = _get_client()
    response = await client.messages.create(
        model=CEO_MODEL,
        max_tokens=500,
        system=_NICHE_ANALYSIS_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Company: {company_name}\n\nDescription: {description}",
        }],
    )
    raw = response.content[0].text
    raw = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n```\s*$", "", raw, flags=re.MULTILINE)

    result = json.loads(raw)
    if result.get("economics_model") not in ("physical", "subscription", "service"):
        result["economics_model"] = "physical"
    return result


_NICHE_SPEC_SYSTEM = """You are a simulation configuration generator. Given a business niche,
generate ONLY the niche-specific parts of the config as JSON.

If a FULL BUSINESS DESCRIPTION is provided, use the founder's actual numbers for pricing,
costs, team structure, and operations. For example:
- If they say "we charge $150/hour", set price_per_unit to 150.0
- If they mention "rent is $8,000/month", set location_annual_cost to 96000
- If they describe specific team roles, map those to corporate_nodes
- If they mention specific suppliers or partners, map those to external_nodes
- If they describe revenue streams, map those to revenue_nodes

You must generate EXACTLY this JSON structure (no markdown, no explanation):
{
  "meta_name": "Human-readable industry name",
  "meta_description": "One sentence description of the simulation",
  "meta_icon": "lucide-icon-name",
  "location_type_key": "snake_case_key",
  "location_type_label": "Human Label",
  "founder_type_key": "snake_case_key",
  "founder_type_label": "Human Label",
  "supplier_types": [
    {"key": "snake_case_key", "label": "Human Label", "cost_modifier_key": "modifier_key", "cost_modifier_value": -0.10}
  ],
  "location_annual_cost": 120000,
  "location_label_for_numbering": "Location",
  "supply_unit_name": "units",
  "max_capacity_per_location": 80,
  "price_per_unit": 50.0,
  "variable_cost_per_unit": 5.0,
  "daily_fixed_costs": 300.0,
  "supply_cost_per_unit": 10.0,
  "price_unit_label": "per unit",
  "cost_unit_label": "cost per unit",
  "starting_cash": 50000,
  "location_open_cost": 50000,
  "max_locations_per_year_cap": 12,
  "variable_cost_modifier_key": "operating_cost",
  "stages_by_location_count": [1, 2, 11, 51],
  "corporate_nodes": [
    {"key": "snake_case", "label": "Human Label", "stage": 2, "annual_cost": 85000,
     "revenue_modifier_key": null, "revenue_modifier_value": null,
     "cost_modifier_key": null, "cost_modifier_value": null}
  ],
  "external_nodes": [
    {"key": "snake_case", "label": "Human Label", "stage": 2,
     "cost_modifier_key": null, "cost_modifier_value": null}
  ],
  "revenue_nodes": [
    {"key": "snake_case", "label": "Human Label", "stage": 2,
     "revenue_modifier_key": "key", "revenue_modifier_value": 0.10}
  ],
  "triggers": [
    {"node_type": "snake_case_key", "label": "Event description",
     "condition_type": "monthly_revenue", "condition_op": ">", "condition_value": 15000}
  ],
  "marketing_contributors": {"node_key": 12.0},
  "infrastructure_multipliers": {"node_key": 1.15},
  "stage_labels": {"1": "Phase 1", "2": "Phase 2", "3": "Phase 3", "4": "Phase 4"}
}

RULES:
- corporate_nodes: 4-8 nodes (key hires, departments as company grows)
- external_nodes: 2-4 nodes (suppliers, partners, investors)
- revenue_nodes: 2-3 nodes (additional revenue streams)
- triggers: one per non-location non-founder non-supplier node. Use condition_type from:
  "monthly_revenue" (with ">" op), "location_count" (with ">=" op), "avg_satisfaction" (with ">" op)
- For compound conditions use "condition_type": "all" and "condition_items": [{"type": "...", "op": "...", "value": ...}]
- stage 1 = starter nodes, stage 2 = early growth, stage 3 = scaling, stage 4 = enterprise
- modifier keys should be descriptive snake_case (e.g. "tech_efficiency", "brand_value")
- variable_cost_modifier_key MUST match one of the cost_modifier_key values you set on nodes;
  it is the key whose multiplier carries volume discounts at scale
- starting_cash should be roughly 4-8 months of single-location operating costs
- location_open_cost should reflect realistic CapEx for opening one unit
- stages_by_location_count: thresholds when company moves to next stage; 4 ascending integers
  (e.g. [1, 2, 11, 51] for retail-style scaling, [1, 3, 10, 30] for B2B SaaS regions)
- max_locations_per_year_cap: realistic per-year expansion ceiling for the industry
- Set null for modifier fields when a node has no modifiers
- All keys must be unique lowercase_with_underscores"""


def _build_spec_from_niche_json(niche_json: dict, economics_model: str) -> dict:
    """Assemble a full IndustrySpec dict from compact niche JSON + economics defaults."""
    nj = niche_json

    # Build nodes dict
    nodes: dict[str, dict] = {}

    # Location node
    loc_key = nj["location_type_key"]
    nodes[loc_key] = {
        "label": nj["location_type_label"],
        "category": "location",
        "stage": 1,
        "annual_cost": nj.get("location_annual_cost", 120000),
    }

    # Founder node
    founder_key = nj["founder_type_key"]
    nodes[founder_key] = {
        "label": nj["founder_type_label"],
        "category": "corporate",
        "stage": 1,
    }

    # Supplier nodes
    supplier_keys = []
    for sup in nj.get("supplier_types", []):
        entry: dict = {
            "label": sup["label"],
            "category": "external",
            "stage": 1,
        }
        if sup.get("cost_modifier_key") and sup.get("cost_modifier_value") is not None:
            entry["cost_modifiers"] = {
                sup["cost_modifier_key"]: sup["cost_modifier_value"]
            }
        nodes[sup["key"]] = entry
        supplier_keys.append(sup["key"])

    # Corporate, external, revenue nodes
    for node in nj.get("corporate_nodes", []):
        entry = {
            "label": node["label"],
            "category": "corporate",
            "stage": node.get("stage", 2),
            "annual_cost": node.get("annual_cost", 60000),
        }
        if node.get("revenue_modifier_key") and node.get("revenue_modifier_value") is not None:
            entry["revenue_modifiers"] = {node["revenue_modifier_key"]: node["revenue_modifier_value"]}
        if node.get("cost_modifier_key") and node.get("cost_modifier_value") is not None:
            entry["cost_modifiers"] = {node["cost_modifier_key"]: node["cost_modifier_value"]}
        nodes[node["key"]] = entry

    for node in nj.get("external_nodes", []):
        entry = {
            "label": node["label"],
            "category": "external",
            "stage": node.get("stage", 2),
        }
        if node.get("cost_modifier_key") and node.get("cost_modifier_value") is not None:
            entry["cost_modifiers"] = {node["cost_modifier_key"]: node["cost_modifier_value"]}
        nodes[node["key"]] = entry

    for node in nj.get("revenue_nodes", []):
        entry = {
            "label": node["label"],
            "category": "revenue",
            "stage": node.get("stage", 2),
        }
        if node.get("revenue_modifier_key") and node.get("revenue_modifier_value") is not None:
            entry["revenue_modifiers"] = {node["revenue_modifier_key"]: node["revenue_modifier_value"]}
        nodes[node["key"]] = entry

    # Scale-relative monthly revenue threshold for opening a new location:
    # 4x the single-location monthly fixed cost burn — proves operations are
    # paying for themselves before doubling up. Avoids hardcoded restaurant $30k.
    daily_fixed = float(nj.get("daily_fixed_costs", 300.0))
    monthly_expansion_threshold = round(daily_fixed * 30 * 4)

    # Build triggers -- location expansion first
    triggers: list[dict] = [{
        "node_type": loc_key,
        "label": f"Opened New {nj['location_type_label']}",
        "is_location_expansion": True,
        "cooldown_ticks": 90,
        "condition": {"monthly_revenue": {">": monthly_expansion_threshold}},
    }]

    for trig in nj.get("triggers", []):
        condition: dict
        if trig.get("condition_type") == "all":
            items = trig.get("condition_items", [])
            condition = {"all": [{item["type"]: {item["op"]: item["value"]}} for item in items]}
        else:
            condition = {trig["condition_type"]: {trig["condition_op"]: trig["condition_value"]}}
        triggers.append({
            "node_type": trig["node_type"],
            "label": trig["label"],
            "condition": condition,
        })

    # Collect quality modifier keys from revenue_modifiers across all nodes
    quality_keys = ["satisfaction_baseline", "ceo_quality_boost"]
    for node_def in nodes.values():
        for k in node_def.get("revenue_modifiers", {}):
            if k not in quality_keys:
                quality_keys.append(k)

    # Count categories
    cat_counts: dict[str, int] = {}
    cat_map = {"location": "Locations", "corporate": "Corporate", "external": "External", "revenue": "Revenue"}
    for nd in nodes.values():
        cat = cat_map.get(nd["category"], nd["category"])
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Variable cost modifier key — AI nominates explicitly. Fall back to first
    # node-declared cost modifier ONLY if no nomination, or to a generic literal.
    declared_cost_keys = sorted({
        k for nd in nodes.values() for k in nd.get("cost_modifiers", {})
    })
    var_cost_key = nj.get("variable_cost_modifier_key") or (
        declared_cost_keys[0] if declared_cost_keys else "operating_cost"
    )

    # Economics-model-specific location defaults
    loc_defaults: dict = {
        "economics_model": economics_model,
        "supply_unit_name": nj.get("supply_unit_name", "units"),
        "location_label": nj.get("location_type_label", "Location"),
        "price": nj.get("price_per_unit", 50.0),
        "max_capacity": nj.get("max_capacity_per_location", 100),
        "variable_cost_per_unit": nj.get("variable_cost_per_unit", 5.0),
        "daily_fixed_costs": daily_fixed,
        "supply_cost_per_unit": nj.get("supply_cost_per_unit", 10.0),
    }
    if economics_model == "subscription":
        loc_defaults.update({
            "capacity_decay_rate": 0.0,
            "churn_rate": 0.03,
            "acquisition_cost": 200.0,
            "scaling_cost_per_unit": 5.0,
        })
    elif economics_model == "service":
        loc_defaults.update({
            "capacity_decay_rate": 0.01,
        })

    # Stage labels
    stage_labels = nj.get("stage_labels", {})
    stage_labels = {int(k): v for k, v in stage_labels.items()}
    if not stage_labels:
        stage_labels = {1: "Startup", 2: "Growth", 3: "Scaling", 4: "Enterprise"}

    # Stage thresholds (minimum location counts to enter each stage)
    stage_thresholds = nj.get("stages_by_location_count", [1, 2, 11, 51])
    if len(stage_thresholds) != 4:
        stage_thresholds = [1, 2, 11, 51]

    example_nodes = [nodes[k]["label"] for k in list(nodes.keys())[:4]]

    slug = nj.get("meta_name", "custom").lower().replace(" ", "_").replace("/", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)

    spec = {
        "meta": {
            "slug": slug,
            "name": nj.get("meta_name", "Custom Industry"),
            "description": nj.get("meta_description", "Custom simulation"),
            "icon": nj.get("meta_icon", "building"),
            "playable": True,
            "total_nodes": len(nodes),
            "growth_stages": 4,
            "key_metrics": ["Daily Revenue", "Avg Satisfaction", "Total Locations"],
            "example_nodes": example_nodes,
            "categories": cat_counts,
        },
        "roles": {
            "location_type": loc_key,
            "founder_type": founder_key,
            "supplier_types": supplier_keys,
            "numbered_labels": {
                loc_key: nj.get("location_label_for_numbering", "Location"),
            },
        },
        "nodes": nodes,
        "triggers": triggers,
        "bridge": {
            "marketing_baseline": 5.0,
            "marketing_per_location": 1.0,
            "sustainable_utilization": 0.85,
            "marketing_contributions": nj.get("marketing_contributors", {}),
            "quality_modifier_keys": quality_keys,
            "infrastructure_multipliers": nj.get("infrastructure_multipliers", {}),
        },
        "constants": {
            "location_open_cost": nj.get("location_open_cost", 50000),
            "employees_per_location": 10,
            "starting_cash": nj.get("starting_cash", 50000),
            "new_location_starting_customers": 20.0,
            "new_location_starting_satisfaction": 0.5,
            "volume_discounts": [[3, 0.90], [1, 1.00]],
            "variable_cost_modifier_key": var_cost_key,
        },
        "stages": [
            {"min_locations": stage_thresholds[0], "stage": 1},
            {"min_locations": stage_thresholds[1], "stage": 2},
            {"min_locations": stage_thresholds[2], "stage": 3},
            {"min_locations": stage_thresholds[3], "stage": 4},
        ],
        "location_defaults": loc_defaults,
        "ceo": {
            "interval_ticks": 182,
            "price_min": round(nj.get("price_per_unit", 50.0) * 0.6, 2),
            "price_max": round(nj.get("price_per_unit", 50.0) * 1.6, 2),
            "price_default": nj.get("price_per_unit", 50.0),
            "cost_min": round(nj.get("variable_cost_per_unit", 5.0) * 0.6, 2),
            "cost_max": round(nj.get("variable_cost_per_unit", 5.0) * 1.6, 2),
            "cost_default": nj.get("variable_cost_per_unit", 5.0),
            "max_locations_per_year_cap": nj.get("max_locations_per_year_cap", 12),
            "price_unit": nj.get("price_unit_label", "per unit"),
            "cost_unit": nj.get("cost_unit_label", "cost per unit"),
            "expansion_overrides": {
                "aggressive": {"cooldown_ticks": 45, "cash_threshold": 60000},
                "normal": {"cooldown_ticks": 90, "cash_threshold": 80000},
                "conservative": {"cooldown_ticks": 180, "cash_threshold": 120000},
            },
        },
        "display": {
            "stage_labels": stage_labels,
            # Generated YAMLs ship empty filters — operator events flow through
            # unfiltered. The frontend can layer industry-specific filters later.
            "event_noise_filters": [],
            "duration_options": [1, 5, 10, 20],
        },
    }
    return spec


async def generate_spec_from_niche(
    niche: str, description: str, economics_model: str, full_description: str = ""
) -> dict:
    """Generate a complete IndustrySpec from a niche description (no interview needed).

    Uses a two-step approach:
    1. Claude generates compact niche-specific JSON (nodes, triggers, labels)
    2. We assemble the full IndustrySpec programmatically with sensible defaults
    """
    full_desc_block = ""
    if full_description.strip():
        full_desc_block = f"\n\nFULL BUSINESS DESCRIPTION (from the founder — use this for accurate pricing, costs, and structure):\n{full_description}"

    user_prompt = f"""Generate the niche-specific configuration for this business:

BUSINESS NICHE: {niche}
BUSINESS SUMMARY: {description}
ECONOMICS MODEL: {economics_model}{full_desc_block}

Generate the JSON now. Remember:
- 4-8 corporate nodes, 2-4 external nodes, 2-3 revenue nodes
- One trigger per non-starter node
- Use the founder's actual pricing, costs, team structure, and revenue details when provided
- Realistic costs and pricing calibrated to this specific business
- All keys must be unique snake_case"""

    client = _get_client()
    last_error = ""

    for attempt in range(3):
        try:
            retry_hint = ""
            if attempt > 0 and last_error:
                retry_hint = f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n{last_error}\nFix the issue and regenerate."

            response = await client.messages.create(
                model=CEO_MODEL,
                max_tokens=3000,
                system=_NICHE_SPEC_SYSTEM,
                messages=[{"role": "user", "content": user_prompt + retry_hint}],
            )
            raw = response.content[0].text

            raw = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\n```\s*$", "", raw, flags=re.MULTILINE)

            niche_json = json.loads(raw)

            # Assemble full spec from niche JSON + defaults
            spec_dict = _build_spec_from_niche_json(niche_json, economics_model)

            # Validate with Pydantic
            spec = IndustrySpec(**spec_dict)

            # Cross-reference validation
            if spec.roles.location_type not in spec.nodes:
                raise ValueError(f"roles.location_type '{spec.roles.location_type}' not in nodes")
            if spec.roles.founder_type not in spec.nodes:
                raise ValueError(f"roles.founder_type '{spec.roles.founder_type}' not in nodes")
            for st in spec.roles.supplier_types:
                if st not in spec.nodes:
                    raise ValueError(f"supplier_type '{st}' not in nodes")
            for trigger in spec.triggers:
                if trigger.node_type not in spec.nodes:
                    raise ValueError(f"trigger node_type '{trigger.node_type}' not in nodes")
            expansion = [t for t in spec.triggers if t.is_location_expansion]
            if len(expansion) != 1:
                raise ValueError(f"Expected exactly 1 expansion trigger, got {len(expansion)}")

            return spec_dict

        except Exception as e:
            last_error = str(e)
            log.warning("Niche spec generation attempt %d/3 failed: %s", attempt + 1, last_error)

    raise ValueError(f"Failed to generate valid spec from niche after 3 attempts: {last_error}")


async def generate_competitor_names(niche: str, num_competitors: int) -> list[str]:
    """Generate niche-appropriate competitor company names via Claude."""
    client = _get_client()
    response = await client.messages.create(
        model=CEO_MODEL,
        max_tokens=200,
        system=(
            "Generate realistic, fictional company names for businesses in the given niche. "
            "Names should sound like real companies -- use a mix of styles (founder names, "
            "descriptive names, abstract/modern names). Respond with ONLY a JSON array of "
            "strings, no explanation. Example: [\"Summit Roasters\", \"Pacific Bean Co\"]"
        ),
        messages=[{
            "role": "user",
            "content": f"Niche: {niche}\nGenerate exactly {num_competitors} company names.",
        }],
    )
    raw = response.content[0].text
    raw = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n```\s*$", "", raw, flags=re.MULTILINE)

    names = json.loads(raw)
    if not isinstance(names, list) or len(names) < num_competitors:
        # Fallback: pad with generic names
        from src.simulation.market.engine import AGENT_NAMES
        while len(names) < num_competitors:
            names.append(AGENT_NAMES[len(names) % len(AGENT_NAMES)])
    return names[:num_competitors]


# ── Document upload + extraction ──

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILES_PER_SESSION = 5


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract text from an uploaded file based on extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if lower.endswith(".txt") or lower.endswith(".csv") or lower.endswith(".md"):
        return file_bytes.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {filename}. Supported: .pdf, .txt, .csv, .md")


async def summarize_document(filename: str, text: str) -> str:
    """Send extracted text to Claude for business-relevant summarization."""
    # Truncate very long documents to fit in context
    if len(text) > 50_000:
        text = text[:50_000] + "\n\n[... truncated ...]"

    client = _get_client()
    response = await client.messages.create(
        model=CEO_MODEL,
        max_tokens=800,
        system=(
            "Extract business-relevant information from this document. Focus on: "
            "revenue model, pricing, cost structure, organizational structure, "
            "growth metrics, competitive landscape, market size, and any financial data. "
            "Be concise -- bullet points preferred. If the document is not business-related, "
            "say so briefly."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Document: {filename}\n\n{text}",
            }
        ],
    )
    return response.content[0].text


async def process_upload(
    session: InterviewSession, filename: str, file_bytes: bytes
) -> dict:
    """Process an uploaded file: extract text, summarize, store in session.

    Returns: {filename, summary}
    """
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {len(file_bytes)} bytes (max {MAX_FILE_SIZE})")
    if len(session.document_summaries) >= MAX_FILES_PER_SESSION:
        raise ValueError(f"Max {MAX_FILES_PER_SESSION} files per session")

    text = extract_text(filename, file_bytes)
    if not text.strip():
        raise ValueError("No text could be extracted from the file")

    summary = await summarize_document(filename, text)

    doc_entry = {"filename": filename, "summary": summary}
    session.document_summaries.append(doc_entry)
    return doc_entry
