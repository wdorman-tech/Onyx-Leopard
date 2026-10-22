"""End-of-run CEO reports: one first-person post-mortem per company.

Written by the run's AI model when AI CEOs are on and budget remains;
otherwise (or on any failure) a plain heuristic summary from final state.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.simulation.engine import DAYS_PER_MONTH, Company
from src.simulation.llm import LlmError, OpenRouterClient

log = logging.getLogger(__name__)

#: Decisions sampled from each end of the run for the report prompt.
DECISION_SAMPLE_EACH_END: int = 10
REASONING_CHARS: int = 200


class CeoReport(BaseModel):
    company_id: str
    company_name: str
    strategy: str
    source: Literal["ai", "heuristic"]
    performance_summary: str
    what_went_well: str
    what_went_wrong: str
    key_decisions: list[str] = Field(default_factory=list)
    final_assessment: str


_REPORT_FIELDS = (
    "performance_summary",
    "what_went_well",
    "what_went_wrong",
    "key_decisions",
    "final_assessment",
)

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_REPORT_FIELDS),
    "properties": {
        "performance_summary": {"type": "string", "description": "2-4 sentences: the arc."},
        "what_went_well": {"type": "string"},
        "what_went_wrong": {"type": "string"},
        "key_decisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-6 inflection points, e.g. 'Day 47: hired a CFO during a crunch'.",
        },
        "final_assessment": {"type": "string", "description": "1-2 sentence verdict."},
    },
}


async def write_report(company: Company, llm: OpenRouterClient | None) -> CeoReport:
    if llm is not None and llm.available:
        try:
            raw = await llm.complete_json(
                system=_system_prompt(company),
                user=_user_prompt(company),
                schema_name="ceo_report",
                schema=REPORT_SCHEMA,
            )
            fields = {k: raw.get(k) for k in _REPORT_FIELDS}
            if isinstance(fields["key_decisions"], str):
                fields["key_decisions"] = [
                    line.strip("-• ").strip() for line in fields["key_decisions"].splitlines()
                ]
            return CeoReport(**_identity(company), source="ai", **fields)
        except (LlmError, ValidationError) as exc:
            log.warning("AI report for %s failed (%s); using heuristic", company.name, exc)
    return heuristic_report(company)


def _identity(company: Company) -> dict[str, str]:
    return {
        "company_id": company.company_id,
        "company_name": company.name,
        "strategy": company.stance.archetype,
    }


def _system_prompt(company: Company) -> str:
    s = company.stance
    return (
        f"You are the CEO of {company.name} ({s.archetype}) writing an end-of-run reflection. "
        f'Your voice: "{s.voice}" Speak in first person. Be honest: survivors acknowledge '
        "near-misses, casualties name the cause without theatre. Cite concrete decisions "
        "and numbers."
    )


def _user_prompt(company: Company) -> str:
    log_ = company.decisions
    sample = (
        log_
        if len(log_) <= 2 * DECISION_SAMPLE_EACH_END
        else (log_[:DECISION_SAMPLE_EACH_END] + log_[-DECISION_SAMPLE_EACH_END:])
    )
    seed = company.seed
    return json.dumps(
        {
            "company": {
                "niche": seed.niche,
                "economics_model": seed.economics_model,
                "starting_cash": round(seed.starting_cash),
                "starting_employees": seed.starting_employees,
                "starting_price": seed.starting_price,
            },
            "final": {
                "alive": company.alive,
                "cash": round(company.cash),
                "monthly_revenue": round(company.daily_revenue * DAYS_PER_MONTH),
                "market_share": round(company.market_share, 4),
                "satisfaction": round(company.satisfaction, 3),
                "employees": company.employees,
                "locations": company.locations,
                "price": round(company.price, 2),
            },
            "monthly_history": [vars(s) for s in company.history],
            "decisions": [
                {
                    "day": d.day,
                    "by": d.source,
                    "spawn": d.spawn,
                    "retire": d.retire,
                    "open_locations": d.open_locations,
                    "price": d.price,
                    "reasoning": d.reasoning[:REASONING_CHARS],
                }
                for d in sample
            ],
            "total_decisions": len(log_),
        },
        separators=(",", ":"),
        default=float,
    )


def heuristic_report(company: Company) -> CeoReport:
    monthly = company.daily_revenue * DAYS_PER_MONTH
    archetype = company.stance.archetype.replace("_", " ")
    if company.alive:
        summary = (
            f"{company.name} survived with ${company.cash:,.0f} cash, ${monthly:,.0f}/mo revenue "
            f"and {company.market_share:.1%} market share."
        )
        well = f"Stayed solvent with {company.locations} locations and {company.employees} staff."
        wrong = "No collapse; a detailed narrative needs the AI CEO path."
        verdict = f"The {archetype} stance held a defensible position."
    else:
        summary = f"{company.name} went bankrupt with ${company.cash:,.0f} cash."
        well = f"Reached {company.locations} locations before cash ran out."
        wrong = "Burn outran revenue for a month straight."
        verdict = f"The {archetype} stance could not stabilize unit economics in time."
    log_ = company.decisions
    picks = log_ if len(log_) <= 6 else log_[:3] + log_[-3:]
    return CeoReport(
        **_identity(company),
        source="heuristic",
        performance_summary=summary,
        what_went_well=well,
        what_went_wrong=wrong,
        key_decisions=[f"Day {d.day}: {d.reasoning[:140]}" for d in picks]
        or ["No CEO decisions were recorded."],
        final_assessment=verdict,
    )
