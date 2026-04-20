"""AI CEO agent layer — Claude-powered strategic decisions on top of the math model.

Each company can have a CEO agent that is called every 6 months of sim time.
The agent reads the company and competitor state, then returns structured
strategic decisions that get applied to the engine state.
"""

from __future__ import annotations

import json
import logging
import os
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import anthropic
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.simulation.config_loader import IndustrySpec
    from src.simulation.unified import CompanyAgent
    from src.simulation.unified_models import UnifiedParams

log = logging.getLogger(__name__)

_DEFAULT_CEO_MODEL = "claude-sonnet-4-6"


# ── Strategy definitions ──

class CEOStrategy(StrEnum):
    AGGRESSIVE_GROWTH = "aggressive_growth"
    QUALITY_FOCUS = "quality_focus"
    COST_LEADER = "cost_leader"
    BALANCED = "balanced"
    MARKET_DOMINATOR = "market_dominator"
    SURVIVOR = "survivor"


STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "aggressive_growth": (
        "Prioritize rapid expansion. Open locations as fast as cash allows. "
        "Accept lower margins for market share. Spend heavily on marketing."
    ),
    "quality_focus": (
        "Prioritize customer satisfaction and product quality above all. "
        "Invest in quality even at higher costs. Expand slowly and deliberately."
    ),
    "cost_leader": (
        "Minimize costs at every opportunity. Keep prices competitive. "
        "Focus on operational efficiency. Expand when margins are strong."
    ),
    "balanced": (
        "Maintain a balanced approach. Grow steadily while maintaining quality. "
        "Don't overextend on any dimension."
    ),
    "market_dominator": (
        "Pursue market share dominance aggressively. Undercut competitors on "
        "price when possible. Invest heavily in marketing. Scale fast."
    ),
    "survivor": (
        "Prioritize survival and cash preservation. Be conservative with "
        "expansion. Maintain strong margins. Weather downturns."
    ),
}


# ── Decision / Report models ──

class CEODecision(BaseModel):
    """Structured output from one CEO agent call.

    Numeric fields are required — there are no industry-agnostic defaults to fall
    back on. Callers (heuristic agent, API fallback) must source defaults from
    the per-industry CeoConfig (price_default, cost_default, etc.).
    """

    reasoning: str
    price_adjustment: float
    expansion_pace: Literal["aggressive", "normal", "conservative"] = "normal"
    marketing_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_investment: float = Field(default=0.0, ge=-0.05, le=0.10)
    cost_target: float
    max_locations_per_year: int
    # CriticAgent metadata (populated when critic mode is used)
    critic_score: float | None = None
    risk_profile: str | None = None


class CEOReport(BaseModel):
    """End-of-simulation report from one CEO agent."""

    company_name: str
    strategy: str
    performance_summary: str
    what_went_well: str
    what_went_wrong: str
    key_decisions: list[str]
    final_assessment: str


class InterviewQA(BaseModel):
    """A single question-answer pair in a CEO interview."""

    question: str
    answer: str


class CEOInterview(BaseModel):
    """Interview transcript from a CEO agent — triggered on bankruptcy or sim end."""

    company_name: str
    strategy: str
    trigger: Literal["bankruptcy", "end_of_simulation"]
    trigger_tick: int
    trigger_year: float
    alive: bool
    final_cash: float
    final_locations: int
    final_market_share: float
    qa_pairs: list[InterviewQA]


# ── Prompt builders ──


def _build_decision_schema(ceo: "CeoConfig") -> str:
    from src.simulation.config_loader import CeoConfig  # noqa: F811

    return f"""{{
  "reasoning": "1-2 sentence explanation of your decision",
  "price_adjustment": {ceo.price_default},       // {ceo.price_unit} (${ceo.price_min}-${ceo.price_max})
  "expansion_pace": "normal",     // "aggressive" | "normal" | "conservative"
  "marketing_intensity": 0.5,     // 0.0 (cut) to 1.0 (double down)
  "quality_investment": 0.0,      // -0.05 (cut) to 0.10 (heavy invest)
  "cost_target": {ceo.cost_default},              // {ceo.cost_unit} (${ceo.cost_min}-${ceo.cost_max})
  "max_locations_per_year": {ceo.max_locations_per_year_cap // 2}     // 0 (freeze) to {ceo.max_locations_per_year_cap}
}}"""


def build_ceo_system_prompt(
    strategy: str,
    spec: IndustrySpec | None = None,
    params: UnifiedParams | None = None,
) -> str:
    """Build the CEO agent system prompt.

    Industry-specific values (price units, ranges, market formula exponents,
    death thresholds) are interpolated from spec/params so the prompt never
    contains restaurant-shaped numbers like $14 or "per plate".
    """
    from src.simulation.config_loader import CeoConfig
    from src.simulation.unified_models import UnifiedParams

    desc = STRATEGY_DESCRIPTIONS.get(strategy, STRATEGY_DESCRIPTIONS["balanced"])
    ceo = spec.ceo if spec else CeoConfig()
    industry_name = spec.meta.name if spec else "Generic Business"
    industry_desc = spec.meta.description if spec else "a generic business"
    p = params if params is not None else UnifiedParams()
    schema = _build_decision_schema(ceo)
    return f"""You are the CEO of a {industry_name} company competing in a shared market simulation.
Industry: {industry_desc}

Your strategy is: **{strategy.replace("_", " ").title()}**
{desc}

Every 6 months of simulated time, you review your company's performance
and competitors, then make strategic decisions for the next 6 months.

Your decisions directly affect operations:
- **price_adjustment** (${ceo.price_min}-${ceo.price_max}): Revenue {ceo.price_unit}. Higher prices
  mean more revenue per unit but may reduce competitiveness.
- **expansion_pace** (aggressive/normal/conservative): How fast to open
  new locations. Aggressive = shorter cooldowns, lower cash thresholds.
- **marketing_intensity** (0.0-1.0): Marketing spend. 0.5 = maintain,
  1.0 = double down, 0.0 = cut entirely.
- **quality_investment** (-0.05 to 0.10): Quality/satisfaction
  investment. Costs more but improves market share via quality score.
- **cost_target** (${ceo.cost_min}-${ceo.cost_max}): {ceo.cost_unit.capitalize()}. Lower = better
  margins, but cutting too deep damages quality.
- **max_locations_per_year** (0-{ceo.max_locations_per_year_cap}): Cap on new locations per year.
  0 = freeze expansion entirely.

Market share is determined by: quality^{p.beta} * marketing^{p.alpha} (multinomial logit).
Companies die if cash stays below ${p.b_death:,.0f} for {p.t_death} consecutive days.

Respond with ONLY a JSON object matching this schema:
{schema}"""


def build_ceo_user_prompt(
    company: CompanyAgent,
    all_companies: list[CompanyAgent],
    tick: int,
    tam: float,
) -> str:
    ceo = company.spec.ceo
    tpy = company.spec.constants.ticks_per_year
    year = tick / tpy
    half = "H1" if (tick % tpy) < (tpy // 2) else "H2"

    # Current price from first location
    current_price = ceo.price_default
    current_unit_cost = ceo.cost_default
    for node in company.state.nodes.values():
        if node.location_state is not None:
            current_price = node.location_state.price
            current_unit_cost = node.location_state.variable_cost_per_unit
            break

    lines = [
        f"SIMULATION DATE: Year {year:.1f}, {half}",
        f"TICK: {tick}",
        "",
        f"YOUR COMPANY ({company.state.name}):",
        f"  Cash: ${company.state.cash:,.0f}",
        f"  Locations: {company.location_count()}",
        f"  Daily Revenue: ${company.daily_revenue:,.0f}",
        f"  Daily Costs: ${company.daily_costs:,.0f}",
        f"  Market Share: {company.share * 100:.1f}%",
        f"  Avg Satisfaction: {company.avg_satisfaction():.1%}",
        f"  Quality Score: {company.quality:.3f}",
        f"  Marketing Score: {company.marketing:.1f}",
        f"  Stage: {company.state.stage}",
        f"  Current Price: ${current_price:.2f}",
        f"  Current {ceo.cost_unit.title()}: ${current_unit_cost:.2f}",
        f"  Total Employees: {company.state.total_employees}",
        "",
        "COMPETITORS:",
    ]

    for c in all_companies:
        if c.state.name == company.state.name:
            continue
        status = "alive" if c.alive else "BANKRUPT"
        lines.append(
            f"  {c.state.name}: {c.location_count()} locs, "
            f"{c.share * 100:.1f}% share, ${c.daily_revenue:,.0f}/day, {status}"
        )

    lines.append(f"\nTotal Addressable Market: ${tam:,.0f}/day")

    # Use persistent agent memory if available, fall back to legacy history
    memory_ctx = company.memory.build_prompt_context(tpy)
    if memory_ctx:
        lines.append("")
        lines.append(memory_ctx)
    elif company._ceo_decision_history:
        lines.append("\nYOUR PREVIOUS DECISIONS:")
        for prev in company._ceo_decision_history[-2:]:
            lines.append(f"  Tick {prev['tick']}: {prev['reasoning']}")

    lines.append(
        f"\nGiven your {company.strategy} strategy, "
        "what are your decisions for the next 6 months?"
    )
    return "\n".join(lines)


def build_report_system_prompt(spec: IndustrySpec | None = None) -> str:
    industry_name = spec.meta.name if spec else "Generic Business"
    return f"""You are writing a post-simulation performance report for a {industry_name} company CEO.

Analyze the company's trajectory, strategic decisions, and competitive outcomes.
Be direct and analytical — highlight what worked, what failed, and why.

Respond with ONLY a JSON object matching this schema:
{{
  "company_name": "string",
  "strategy": "string",
  "performance_summary": "2-3 sentence overview of the company's journey",
  "what_went_well": "2-3 sentences on successes",
  "what_went_wrong": "2-3 sentences on failures or missed opportunities",
  "key_decisions": ["list of 3-5 pivotal decisions and their outcomes"],
  "final_assessment": "1-2 sentence overall verdict"
}}"""


def build_report_user_prompt(
    company: CompanyAgent,
    all_companies: list[CompanyAgent],
    tick: int,
    tam: float,
) -> str:
    tpy = company.spec.constants.ticks_per_year
    years = tick / tpy

    lines = [
        f"SIMULATION COMPLETE — {years:.1f} years simulated",
        "",
        f"COMPANY: {company.state.name}",
        f"Strategy: {company.strategy}",
        f"Final Status: {'ALIVE' if company.alive else 'BANKRUPT'}",
        f"Final Cash: ${company.state.cash:,.0f}",
        f"Final Locations: {company.location_count()}",
        f"Final Market Share: {company.share * 100:.1f}%",
        f"Final Daily Revenue: ${company.daily_revenue:,.0f}",
        f"Total Employees: {company.state.total_employees}",
        "",
        "FINAL STANDINGS:",
    ]

    # Sort by share descending
    ranked = sorted(all_companies, key=lambda c: c.share, reverse=True)
    for i, c in enumerate(ranked, 1):
        marker = " <-- YOU" if c.state.name == company.state.name else ""
        status = "alive" if c.alive else "BANKRUPT"
        lines.append(
            f"  {i}. {c.state.name} ({c.strategy or 'autopilot'}): "
            f"{c.share * 100:.1f}% share, {c.location_count()} locs, "
            f"${c.state.cash:,.0f} cash, {status}{marker}"
        )

    if company._ceo_decision_history:
        lines.append(f"\nDECISION HISTORY ({len(company._ceo_decision_history)} decisions):")
        for d in company._ceo_decision_history:
            lines.append(
                f"  Tick {d['tick']} (Year {d['tick']/tpy:.1f}): "
                f"price=${d.get('price_adjustment', 0.0):.2f}, "
                f"expansion={d.get('expansion_pace', 'normal')}, "
                f"marketing={d.get('marketing_intensity', 0.5):.1f}, "
                f"quality={d.get('quality_investment', 0):.2f} — "
                f"{d.get('reasoning', '')}"
            )

    return "\n".join(lines)


# ── Claude API calls ──

def _get_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env


def _parse_json_response(text: str) -> dict:
    """Extract JSON from a response that might have markdown fences or preamble text."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model")

    # Try direct parse first (ideal case: response is pure JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences anywhere in the response
    import re
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1).strip())

    # Last resort: find the outermost { ... } in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"No JSON found in response: {text[:200]}")


async def call_ceo_agent(
    company_name: str,
    system_prompt: str,
    user_prompt: str,
    ceo_config: "CeoConfig",
    retries: int = 2,
    model: str = _DEFAULT_CEO_MODEL,
) -> CEODecision:
    """Call Claude API for a CEO decision. Falls back to industry defaults on failure."""
    client = _get_client()

    for attempt in range(retries + 1):
        raw = ""
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=400,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
            data = _parse_json_response(raw)
            return CEODecision(**data)
        except Exception as e:
            log.warning(
                "CEO agent call failed for %s (attempt %d/%d): %s\nRaw response: %.300s",
                company_name, attempt + 1, retries + 1, e, raw,
            )
            if attempt < retries:
                continue

    log.error("CEO agent failed all retries for %s, using industry defaults", company_name)
    return CEODecision(
        reasoning="Maintaining current course (agent unavailable)",
        price_adjustment=ceo_config.price_default,
        cost_target=ceo_config.cost_default,
        max_locations_per_year=ceo_config.max_locations_per_year_cap // 2,
    )


async def call_report_agent(
    company_name: str,
    system_prompt: str,
    user_prompt: str,
    model: str = _DEFAULT_CEO_MODEL,
) -> CEOReport:
    """Call Claude API for an end-of-simulation report."""
    client = _get_client()

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text
        data = _parse_json_response(raw)
        return CEOReport(**data)
    except Exception as e:
        log.error("Report agent failed for %s: %s", company_name, e)
        return CEOReport(
            company_name=company_name,
            strategy="unknown",
            performance_summary="Report generation failed.",
            what_went_well="Unable to assess.",
            what_went_wrong="Unable to assess.",
            key_decisions=[],
            final_assessment="Report unavailable due to agent error.",
        )


# ── Decision application ──

def apply_decision(company: CompanyAgent, decision: CEODecision, tick: int) -> None:
    """Mutate company state based on a CEO decision."""
    ceo = company.spec.ceo
    loc_type = company.spec.roles.location_type

    # Clamp values to industry-specific ranges
    decision.price_adjustment = max(ceo.price_min, min(ceo.price_max, decision.price_adjustment))
    decision.cost_target = max(ceo.cost_min, min(ceo.cost_max, decision.cost_target))
    decision.max_locations_per_year = max(0, min(ceo.max_locations_per_year_cap, decision.max_locations_per_year))

    # Price adjustment — set on all locations
    for node in company.state.nodes.values():
        if node.location_state is not None:
            node.location_state.price = decision.price_adjustment

    # Food cost target — set on all locations
    for node in company.state.nodes.values():
        if node.location_state is not None:
            node.location_state.variable_cost_per_unit = decision.cost_target

    # Quality investment — adjust satisfaction_baseline modifier on location nodes
    for node in company.state.nodes.values():
        if node.type == loc_type and node.active:
            node.revenue_modifiers["ceo_quality_boost"] = decision.quality_investment

    # Store CEO-controlled parameters on the company
    company.marketing_boost = decision.marketing_intensity
    company.expansion_pace = decision.expansion_pace
    company.max_locations_per_year = decision.max_locations_per_year

    # Record decision in history
    record = decision.model_dump()
    record["tick"] = tick
    company._ceo_decision_history.append(record)

    # Rebuild location arrays since we changed price and food_cost
    company._rebuild_loc_arrays()


def check_api_key() -> bool:
    """Check if the Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def build_critic_system_prompt(spec: IndustrySpec | None = None) -> str:
    """System prompt for the CriticAgent that scores candidate decisions."""
    industry_name = spec.meta.name if spec else "Generic Business"
    return f"""You are a strategic analyst evaluating CEO decisions for a {industry_name} company.

You will receive 3 candidate decisions with different risk profiles:
- AGGRESSIVE: high-growth, high-risk
- MODERATE: balanced approach
- CONSERVATIVE: safety-first, capital preservation

Score each candidate on a 0-10 scale considering:
1. Alignment with the company's current situation (cash, market position)
2. Risk-reward tradeoff given competitive landscape
3. Long-term viability vs. short-term gain

Respond with ONLY a JSON object:
{{
  "scores": [
    {{"profile": "aggressive", "score": 7.5, "rationale": "brief reason"}},
    {{"profile": "moderate", "score": 8.0, "rationale": "brief reason"}},
    {{"profile": "conservative", "score": 6.0, "rationale": "brief reason"}}
  ],
  "selected": "moderate",
  "selection_reasoning": "why this is the best choice given the context"
}}"""


async def call_critic_agent(
    company_name: str,
    system_prompt: str,
    user_prompt: str,
    candidates: list[CEODecision],
    model: str = _DEFAULT_CEO_MODEL,
) -> tuple[CEODecision, dict]:
    """Score multiple candidate decisions and select the best.

    Returns (selected_decision, critic_evaluation).
    Falls back to the moderate candidate on failure.
    """
    client = _get_client()

    # Format candidates for the prompt
    candidate_text = "\n\n".join(
        f"CANDIDATE {i+1} ({c.risk_profile or 'unknown'}):\n"
        f"  reasoning: {c.reasoning}\n"
        f"  price: ${c.price_adjustment:.2f}, expansion: {c.expansion_pace}\n"
        f"  marketing: {c.marketing_intensity:.1f}, quality: {c.quality_investment:.2f}\n"
        f"  cost_target: ${c.cost_target:.2f}, max_locs/yr: {c.max_locations_per_year}"
        for i, c in enumerate(candidates)
    )

    full_prompt = f"{user_prompt}\n\n--- CANDIDATE DECISIONS TO EVALUATE ---\n{candidate_text}"

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": full_prompt}],
        )
        raw = response.content[0].text
        data = _parse_json_response(raw)

        selected_profile = data.get("selected", "moderate")
        evaluation = data

        # Find the selected candidate
        profile_map = {c.risk_profile: c for c in candidates if c.risk_profile}
        selected = profile_map.get(selected_profile, candidates[1])  # default moderate

        # Attach critic score
        for score_entry in data.get("scores", []):
            if score_entry.get("profile") == selected_profile:
                selected.critic_score = score_entry.get("score")
                break

        return selected, evaluation

    except Exception as e:
        log.warning("CriticAgent failed for %s: %s — using moderate candidate", company_name, e)
        return candidates[1] if len(candidates) > 1 else candidates[0], {}


async def validate_api_key() -> tuple[bool, str]:
    """Validate the API key with a minimal Claude call. Returns (ok, error_msg)."""
    if not check_api_key():
        return False, "ANTHROPIC_API_KEY environment variable is not set"

    client = _get_client()
    try:
        await client.messages.create(
            model=_DEFAULT_CEO_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True, ""
    except anthropic.AuthenticationError:
        return False, (
            "ANTHROPIC_API_KEY is invalid or expired"
            " — generate a new key at console.anthropic.com"
        )
    except anthropic.PermissionDeniedError:
        return False, "ANTHROPIC_API_KEY lacks permission to use the required model"
    except Exception as e:
        return False, f"API key validation failed: {e}"
