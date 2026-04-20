"""Per-agent persistent memory system.

Each company's AI agent carries a running memory of its decisions, outcomes,
and compressed summaries. Recent decisions are kept in full detail; older
ones are compressed to a narrative summary for prompt context.
"""

from __future__ import annotations

from enum import StrEnum


class AgentTier(StrEnum):
    """AI agent capability tiers — matched to model cost/capability."""

    EXECUTIVE = "executive"      # Sonnet — strategic decisions
    OPERATIONAL = "operational"  # Haiku — operational decisions
    HEURISTIC = "heuristic"      # Rule-based — no API calls


class AIBudget:
    """Tracks AI API spending within a simulation run.

    When the budget is exhausted, agents fall back to the heuristic tier.
    """

    # Conservative per-call cost estimates (input + output tokens)
    COST_PER_CALL: dict[str, float] = {
        "claude-sonnet-4-6": 0.015,
        "claude-haiku-4-5-20251001": 0.002,
    }

    def __init__(self, max_spend: float = 1.0) -> None:
        self.max_spend = max_spend
        self.total_spent: float = 0.0
        self.call_count: int = 0

    @property
    def exhausted(self) -> bool:
        return self.total_spent >= self.max_spend

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_spend - self.total_spent)

    def can_afford(self, model: str) -> bool:
        if self.exhausted:
            return False
        cost = self.COST_PER_CALL.get(model, 0.01)
        return self.total_spent + cost <= self.max_spend

    def record_call(self, model: str) -> None:
        cost = self.COST_PER_CALL.get(model, 0.01)
        self.total_spent += cost
        self.call_count += 1


class AgentMemory:
    """Persistent memory for a company's AI agent across the simulation.

    Full decision history with automatic compression of older entries.
    Recent decisions (last `recent_window`) are kept in full detail;
    older ones are compressed to a running narrative.
    """

    def __init__(self, recent_window: int = 5) -> None:
        self.decisions: list[dict] = []
        self.summary: str = ""
        self.recent_window = recent_window
        self.total_decisions: int = 0

        # Outcome tracking. min_cash starts as None (no observations yet) rather
        # than float("inf") because inf is not valid JSON and breaks serialization
        # of an empty memory. The first record_decision sets it to a real value.
        self.peak_cash: float = 0.0
        self.min_cash: float | None = None
        self.peak_share: float = 0.0
        self.peak_locations: int = 0
        self.crisis_count: int = 0

    def record_decision(
        self,
        decision_data: dict,
        tick: int,
        cash: float,
        share: float,
        locations: int,
        daily_revenue: float,
    ) -> None:
        """Record a new decision with the company state at decision time."""
        record = {
            "tick": tick,
            "decision": decision_data,
            "cash": cash,
            "share": share,
            "locations": locations,
            "daily_revenue": daily_revenue,
        }
        self.decisions.append(record)
        self.total_decisions += 1

        # Track extremes
        self.peak_cash = max(self.peak_cash, cash)
        self.min_cash = cash if self.min_cash is None else min(self.min_cash, cash)
        self.peak_share = max(self.peak_share, share)
        self.peak_locations = max(self.peak_locations, locations)
        if cash < 0:
            self.crisis_count += 1

        self._compress_if_needed()

    def _compress_if_needed(self) -> None:
        """Move older decisions into a compressed narrative summary."""
        if len(self.decisions) <= self.recent_window:
            return

        to_compress = self.decisions[: -self.recent_window]
        self.decisions = self.decisions[-self.recent_window :]

        parts: list[str] = []
        if self.summary:
            parts.append(self.summary)

        for record in to_compress:
            tick = record["tick"]
            reasoning = record["decision"].get("reasoning", "")
            cash = record["cash"]
            share = record["share"]
            locs = record["locations"]
            parts.append(
                f"Tick {tick}: {reasoning} "
                f"(cash=${cash:,.0f}, share={share * 100:.1f}%, {locs} locs)"
            )

        self.summary = "\n".join(parts)

    def build_prompt_context(self, ticks_per_year: float) -> str:
        """Build a prompt section describing the agent's decision history."""
        lines: list[str] = []

        if self.summary:
            lines.append("EARLIER DECISIONS (COMPRESSED):")
            lines.append(self.summary)
            lines.append("")

        if self.decisions:
            lines.append(
                f"RECENT DECISIONS ({len(self.decisions)} of {self.total_decisions} total):"
            )
            for record in self.decisions:
                tick = record["tick"]
                year = tick / ticks_per_year
                d = record["decision"]
                lines.append(
                    f"  Year {year:.1f} (tick {tick}): {d.get('reasoning', '')}"
                )
                lines.append(
                    f"    -> price=${d.get('price_adjustment', 0.0):.2f}, "
                    f"expansion={d.get('expansion_pace', 'normal')}, "
                    f"marketing={d.get('marketing_intensity', 0.5):.1f}, "
                    f"quality={d.get('quality_investment', 0):.2f}"
                )
                lines.append(
                    f"    State: cash=${record['cash']:,.0f}, "
                    f"share={record['share'] * 100:.1f}%, "
                    f"{record['locations']} locs, "
                    f"${record['daily_revenue']:,.0f}/day revenue"
                )

        if self.total_decisions > 0:
            lines.append("")
            lines.append("TRACK RECORD:")
            lines.append(f"  Peak cash: ${self.peak_cash:,.0f}")
            if self.min_cash is not None:
                lines.append(f"  Lowest cash: ${self.min_cash:,.0f}")
            lines.append(f"  Peak market share: {self.peak_share * 100:.1f}%")
            lines.append(f"  Max locations: {self.peak_locations}")
            if self.crisis_count > 0:
                lines.append(
                    f"  Cash crises (negative cash at decision time): {self.crisis_count}"
                )

        return "\n".join(lines)
