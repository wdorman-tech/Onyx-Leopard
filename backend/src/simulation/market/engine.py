"""Competitive market simulation engine.

Implements the 10-step simulation loop from competitive_market_simulation_spec.md:
Agent-Based Logistic Growth Model with Competitive Share Attraction.
"""

from __future__ import annotations

import math
import random as _random_module

from src.simulation.market.colors import agent_color
from src.simulation.market.models import (
    AgentParams,
    AgentSnapshot,
    AgentState,
    MarketParams,
    PendingExpansion,
)

AGENT_NAMES: list[str] = [
    "Alpha Corp", "Beta Inc", "Gamma Ltd", "Delta Co", "Epsilon Group",
    "Zeta Holdings", "Eta Ventures", "Theta Partners", "Iota Systems", "Kappa Industries",
    "Lambda Tech", "Mu Services", "Nu Dynamics", "Xi Solutions", "Omicron Labs",
    "Pi Global", "Rho Capital", "Sigma Works", "Tau Enterprises", "Upsilon Co",
]


# ── Pure math functions (spec equations, extracted for testability) ──


def compute_share_attraction(
    agents: list[AgentState],
    alpha: float,
    beta: float,
) -> list[float]:
    """Multinomial logit share attraction (spec Section 3.2).

    s_i = (q_i^beta * m_i^alpha) / SUM_j(q_j^beta * m_j^alpha)
    """
    alive = [a for a in agents if a.alive]
    if not alive:
        return [0.0] * len(agents)

    attractions: dict[str, float] = {}
    for a in alive:
        m = max(a.marketing, 0.0)
        q = max(a.quality, 1e-12)
        attractions[a.id] = (q ** beta) * (m ** alpha)

    total = sum(attractions.values())

    shares: list[float] = []
    for a in agents:
        if not a.alive:
            shares.append(0.0)
        elif total <= 0:
            shares.append(1.0 / len(alive))
        else:
            shares.append(attractions[a.id] / total)
    return shares


def compute_capital_constraint(
    cash: float,
    b_threshold: float,
    k_sigmoid: float,
) -> float:
    """Sigmoid capital constraint (spec Section 3.5).

    C_i = 1 / (1 + exp(-k * (B_i - B_threshold)))
    """
    x = -k_sigmoid * (cash - b_threshold)
    # Clamp to avoid overflow
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(x))


def compute_hhi(shares: list[float]) -> float:
    """Herfindahl-Hirschman Index: SUM_i(s_i^2)."""
    return sum(s * s for s in shares if s > 0)


def compute_spawn_probability(
    hhi: float,
    lambda_entry: float,
    g_market: float,
    g_ref: float,
    unserved_ratio: float = 0.0,
) -> float:
    """New entrant spawn probability.

    Two entry channels:
      1. Competition-driven (spec Section 5.2):
         lambda * max(0, 1-HHI) * max(0, g_market) / g_ref
      2. Demand-driven (fixes dead-market problem):
         lambda * unserved_ratio * 0.5
    The demand channel ensures new entrants are attracted even at HHI=1.0
    when most of the TAM is unserved.
    """
    if g_ref <= 0:
        return 0.0
    growth_factor = max(0.0, g_market) / g_ref
    competition_entry = lambda_entry * max(0.0, 1.0 - hhi) * growth_factor
    demand_entry = lambda_entry * max(0.0, min(1.0, unserved_ratio)) * 0.5
    return competition_entry + demand_entry


class MarketEngine:
    """Competitive market simulation engine.

    Manages multiple agents competing for share of a total addressable market.
    Each tick implements the 10-step loop from the spec.
    """

    def __init__(
        self,
        params: MarketParams,
        max_ticks: int = 0,
        seed: int | None = None,
    ) -> None:
        self.params = params
        self.max_ticks = max_ticks
        self.rng = _random_module.Random(seed)

        self.tick_num: int = 0
        self.tam: float = params.tam_0
        self.agents: list[AgentState] = []
        self._next_agent_idx: int = 0
        self._status: str = "operating"

        # Spawn initial agents
        for _ in range(params.n_0):
            self._spawn_agent()

        # Give initial agents a small starting revenue so dynamics kick in
        for agent in self.agents:
            agent.revenue = self.tam / len(self.agents) * self.rng.uniform(0.05, 0.15)

    @property
    def is_complete(self) -> bool:
        if self._status == "collapsed":
            return True
        if self.max_ticks > 0 and self.tick_num >= self.max_ticks:
            return True
        return False

    def _spawn_agent(self) -> AgentState:
        """Create a new agent with randomized parameters."""
        p = self.params
        idx = self._next_agent_idx
        self._next_agent_idx += 1

        name = AGENT_NAMES[idx % len(AGENT_NAMES)]
        if idx >= len(AGENT_NAMES):
            name = f"{name} {idx // len(AGENT_NAMES) + 1}"

        agent_params = AgentParams(
            name=name,
            r=self.rng.uniform(*p.r_range),
            margin=self.rng.uniform(*p.margin_range),
            f=self.rng.uniform(*p.f_range),
            eta_m=self.rng.uniform(*p.eta_m_range),
            eta_q=self.rng.uniform(*p.eta_q_range),
            tau_k=self.rng.randint(*p.tau_k_range),
        )

        quality = self.rng.uniform(*p.q_range)
        marketing = self.rng.uniform(*p.m_range)
        capacity = self.rng.uniform(*p.k_range)
        cash = self.rng.uniform(*p.b_range)

        agent = AgentState(
            id=f"agent-{idx}",
            params=agent_params,
            revenue=0.0,
            cash=cash,
            capacity=capacity,
            quality=quality,
            marketing=marketing,
            q_target=quality,
            m_target=marketing,
            k_target=capacity,
            color=agent_color(idx),
        )
        self.agents.append(agent)
        return agent

    def _alive_agents(self) -> list[AgentState]:
        return [a for a in self.agents if a.alive]

    def _quarterly_review(self, agent: AgentState, tam: float) -> list[str]:
        """Execute the 6 heuristic decision rules.

        Rules 1-5 from spec Section 4.2.1 (thresholds adjusted for linear
        convergence equilibrium).  Rule 6 is a market-opportunity expansion
        that triggers when demand far exceeds capacity.

        Rules execute in order; later rules can override earlier ones.
        """
        events: list[str] = []
        ap = agent.params
        util = agent.revenue / agent.capacity if agent.capacity > 0 else 0.0

        # Rule 1: Cash Emergency
        if agent.cash < 2 * ap.f:
            agent.m_target = agent.marketing * 0.5
            agent.k_target = agent.capacity
            events.append(f"{ap.name}: Cash emergency — cutting marketing")

        # Rule 2: Excess Capacity
        if agent.capacity > 0 and util < 0.5:
            agent.m_target = agent.marketing * 1.2

        # Rule 3: Capacity Constraint  (lowered from 0.85 — linear convergence
        # equilibrium R/K ≈ r*C/(r*C+delta) sits near 0.75, so 0.85 was
        # unreachable; 0.75 lets the rule fire at equilibrium for fast growers)
        if agent.capacity > 0 and util > 0.75:
            agent.k_target = agent.capacity * 1.3
            events.append(f"{ap.name}: At capacity — expanding 30%")

        # Rule 4: Market Share Decline
        if agent.prev_share > 0 and agent.share < agent.prev_share * 0.9:
            agent.q_target = agent.quality * 1.1
            agent.m_target = agent.marketing * 1.15
            events.append(f"{ap.name}: Losing share — investing in quality & marketing")

        # Rule 5: Profitable Growth  (lowered from 0.70 to match achievable
        # equilibrium — ensures agents with cash actually invest)
        if agent.cash > 5 * ap.f and agent.capacity > 0 and util > 0.60:
            agent.q_target = agent.quality * 1.05
            agent.k_target = agent.capacity * 1.15

        # Rule 6: Market Opportunity — expand when demand far exceeds capacity
        # and the agent has a reasonable cash cushion.  Capped at 1.5× per
        # quarter to prevent runaway expansion.
        demand_potential = tam * agent.share
        if (demand_potential > agent.capacity * 1.2
                and agent.cash > 3 * ap.f):
            target_cap = min(demand_potential, agent.capacity * 1.5)
            if target_cap > agent.k_target:
                agent.k_target = target_cap
                events.append(f"{ap.name}: Market opportunity — expanding toward demand")

        return events

    def tick(self) -> dict:  # noqa: C901 — complex by spec design
        """Run one simulation step. Returns tick data dict for SSE."""
        self.tick_num += 1
        p = self.params
        events: list[str] = []
        alive = self._alive_agents()

        if not alive:
            self._status = "collapsed"
            return self._build_result(events)

        # ── Step 1: Update TAM ──
        self.tam *= (1.0 + p.g_market)

        # ── Step 2: Compute share attraction ──
        shares = compute_share_attraction(self.agents, p.alpha, p.beta)
        for agent, share in zip(self.agents, shares):
            agent.share = share

        # ── Step 3: Effective ceiling per agent ──
        ceilings: dict[str, float] = {}
        for agent in alive:
            ceilings[agent.id] = min(self.tam * agent.share, agent.capacity)

        # ── Step 4: Update revenue (linear convergence + churn) ──
        # R += r * max(0, M - R) * C - delta * R
        # Linear convergence: growth proportional to gap between ceiling and
        # current revenue.  Equilibrium R/M = r*C / (r*C + delta).
        for agent in alive:
            m_i = ceilings[agent.id]
            c_i = compute_capital_constraint(agent.cash, p.b_threshold, p.k_sigmoid)

            gap = max(0.0, m_i - agent.revenue)
            growth = agent.params.r * gap * c_i
            churn = p.delta * agent.revenue
            agent.revenue = max(0.0, agent.revenue + growth - churn)

        # ── Step 5: Update cash balance ──
        for agent in alive:
            income = agent.revenue * agent.params.margin
            capacity_cost = p.c_k * agent.capacity
            agent.cash += income - capacity_cost - agent.marketing - agent.params.f

        # ── Step 6: Continuous adjustment (drift toward targets) ──
        for agent in alive:
            ap = agent.params
            agent.marketing += ap.eta_m * (agent.m_target - agent.marketing)
            agent.quality += ap.eta_q * (agent.q_target - agent.quality)
            agent.quality = max(0.01, agent.quality)
            agent.marketing = max(0.0, agent.marketing)

        # ── Step 7: Quarterly review ──
        if self.tick_num % p.t_q == 0:
            for agent in alive:
                review_events = self._quarterly_review(agent, self.tam)
                events.extend(review_events)

                # Commit capacity expansion if target > current
                if agent.k_target > agent.capacity:
                    increment = agent.k_target - agent.capacity
                    capex = increment * p.capex_per_unit
                    agent.cash -= capex
                    agent.pending_expansions.append(
                        PendingExpansion(
                            delivery_tick=self.tick_num + agent.params.tau_k,
                            new_capacity=agent.k_target,
                        )
                    )

                # Store share for next quarterly comparison
                agent.prev_share = agent.share

        # ── Step 8: Capacity delivery ──
        for agent in alive:
            delivered = [
                exp for exp in agent.pending_expansions
                if self.tick_num >= exp.delivery_tick
            ]
            for exp in delivered:
                agent.capacity = max(agent.capacity, exp.new_capacity)
                events.append(f"{agent.params.name}: Capacity expanded to {exp.new_capacity:.0f}")
            agent.pending_expansions = [
                exp for exp in agent.pending_expansions
                if self.tick_num < exp.delivery_tick
            ]

        # ── Step 9: Death check + revenue redistribution ──
        total_freed = 0.0
        for agent in self.agents:
            if not agent.alive:
                # Decay dead agent's revenue — freed portion redistributed
                if agent.decay_ticks_remaining > 0:
                    freed = agent.revenue / agent.decay_ticks_remaining
                    agent.revenue = max(0.0, agent.revenue - freed)
                    total_freed += freed
                    agent.decay_ticks_remaining -= 1
                continue

            if agent.cash < p.b_death:
                agent.death_counter += 1
            else:
                agent.death_counter = 0

            if agent.death_counter >= p.t_death:
                agent.alive = False
                agent.death_counter = 0
                agent.decay_ticks_remaining = p.tau_decay
                events.append(f"{agent.params.name}: BANKRUPT")

        # Redistribute freed revenue to survivors proportional to share,
        # capped by each agent's spare capacity so revenue ≤ capacity.
        alive_now = self._alive_agents()
        if total_freed > 0 and alive_now:
            total_share = sum(a.share for a in alive_now)
            if total_share > 0:
                for agent in alive_now:
                    alloc = total_freed * (agent.share / total_share)
                    room = max(0.0, agent.capacity - agent.revenue)
                    agent.revenue += min(alloc, room)

        # ── Step 10: Spawn check ──
        alive = self._alive_agents()
        if alive:
            alive_shares = [a.share for a in alive]
            hhi = compute_hhi(alive_shares)
            captured = sum(a.revenue for a in alive)
            unserved = max(0.0, 1.0 - captured / self.tam) if self.tam > 0 else 0.0
            p_spawn = compute_spawn_probability(
                hhi, p.lambda_entry, p.g_market, p.g_ref, unserved,
            )
            if self.rng.random() < p_spawn:
                new_agent = self._spawn_agent()
                new_agent.revenue = self.tam * 0.01 * self.rng.uniform(0.5, 1.5)
                events.append(f"{new_agent.params.name}: Entered the market")

        # Check if all dead
        if not self._alive_agents():
            self._status = "collapsed"

        return self._build_result(events)

    def _build_result(self, events: list[str]) -> dict:
        """Build the tick result dict for SSE streaming."""
        alive = self._alive_agents()
        alive_shares = [a.share for a in alive]
        hhi = compute_hhi(alive_shares) if alive else 1.0
        captured = sum(a.revenue for a in self.agents if a.revenue > 0)

        agent_snapshots: list[dict] = []
        for a in self.agents:
            util = a.revenue / a.capacity if a.capacity > 0 else 0.0
            tam_share_ceiling = self.tam * a.share if a.share > 0 else 0.0
            constraint = "capacity" if a.capacity <= tam_share_ceiling else "demand"

            snap = AgentSnapshot(
                id=a.id,
                name=a.params.name,
                alive=a.alive,
                revenue=round(a.revenue, 2),
                cash=round(a.cash, 2),
                capacity=round(a.capacity, 2),
                quality=round(a.quality, 4),
                marketing=round(a.marketing, 2),
                share=round(a.share, 6),
                utilization=round(min(1.0, util), 4),
                binding_constraint=constraint,
                color=a.color,
            )
            agent_snapshots.append(snap.model_dump())

        return {
            "tick": self.tick_num,
            "status": self._status,
            "mode": "market",
            "tam": round(self.tam, 2),
            "captured": round(captured, 2),
            "hhi": round(hhi, 6),
            "agent_count": len(alive),
            "agents": agent_snapshots,
            "events": events,
        }
