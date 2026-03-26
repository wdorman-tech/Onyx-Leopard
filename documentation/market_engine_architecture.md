# Market Engine Architecture

## Overview

The competitive market simulation is a standalone module (`backend/src/simulation/market/`) that models multiple firms competing for market share in a Total Addressable Market (TAM). It runs alongside the existing growth simulation engine, sharing the same session/SSE infrastructure.

---

## Module Structure

```
backend/src/simulation/market/
├── __init__.py          # Package init
├── models.py            # Pydantic data models (MarketParams, AgentState, etc.)
├── engine.py            # Core simulation engine (MarketEngine + pure math functions)
├── presets.py           # 4 preset parameter bundles
└── colors.py            # 20-color palette for agent visualization
```

---

## Core Equations

### Revenue Dynamics (Linear Convergence)

Each agent's revenue converges toward its effective ceiling, with churn pulling it down:

```
R(t+1) = R(t) + r * max(0, M(t) - R(t)) * C(t) - delta * R(t)
```

Where:
- `M(t) = min(TAM * share, capacity)` — effective ceiling (demand or supply bound)
- `r` — demand capture rate (how fast the agent fills unserved demand)
- `C(t)` — sigmoid capital constraint (0 to 1, based on cash health)
- `delta` — per-tick churn rate

**Equilibrium:** R*/M = r*C / (r*C + delta). With r ≈ 3*delta and C ≈ 1, equilibrium utilization ≈ 0.75.

### Share Attraction (Multinomial Logit)

```
s_i = (q_i^beta * m_i^alpha) / SUM_j(q_j^beta * m_j^alpha)
```

- `alpha` controls marketing elasticity
- `beta` controls quality elasticity
- Sum runs over alive agents only

### Capital Constraint (Sigmoid)

```
C(t) = 1 / (1 + exp(-k * (cash - b_threshold)))
```

Maps cash health to a 0-1 growth multiplier. Healthy cash → C ≈ 1. Low cash → C → 0.

### Spawn Probability (Dual Channel)

```
P = lambda * (1-HHI) * g_market/g_ref     [competition-driven]
  + lambda * unserved_ratio * 0.5          [demand-driven]
```

Two independent channels for market entry. The demand channel ensures new firms enter even in monopolized markets with unserved demand.

---

## 10-Step Simulation Loop

Each tick executes these steps in order:

1. **Update TAM** — `TAM *= (1 + g_market)`
2. **Compute share attraction** — multinomial logit for alive agents
3. **Compute effective ceiling** — `M_i = min(TAM * s_i, K_i)` per agent
4. **Update revenue** — linear convergence + churn
5. **Update cash** — income - capacity cost - marketing - fixed costs
6. **Continuous adjustment** — drift marketing/quality toward targets
7. **Quarterly review** — heuristic rules set new targets (every T_q ticks)
8. **Capacity delivery** — pending expansions activate after tau_K delay
9. **Death check + redistribution** — bankruptcy, revenue freed to survivors
10. **Spawn check** — probabilistic new entrant creation

---

## Heuristic Decision Rules (Quarterly)

Six rules execute in order. Later rules override earlier ones.

| # | Rule | Condition | Action |
|---|------|-----------|--------|
| 1 | Cash Emergency | cash < 2*F | Cut marketing 50%, freeze capacity |
| 2 | Excess Capacity | R/K < 0.50 | Boost marketing 20% |
| 3 | Capacity Constraint | R/K > 0.75 | Expand capacity 30% |
| 4 | Share Decline | share < prev_share * 0.9 | Invest in quality + marketing |
| 5 | Profitable Growth | cash > 5*F AND R/K > 0.60 | Moderate expansion 15% |
| 6 | Market Opportunity | demand > K*1.2 AND cash > 3*F | Expand toward demand (max 1.5×/quarter) |

---

## Presets

| Preset | Character | alpha | beta | delta | Firms | Key Dynamic |
|--------|-----------|-------|------|-------|-------|-------------|
| Price War | Marketing-driven | 1.5 | 0.5 | 0.15 | 8 | High spend, high churn, consolidation |
| Innovation Race | Quality-driven | 0.3 | 1.5 | 0.05 | 5 | R&D wins, slow dynamics |
| Monopoly | Balanced | 0.8 | 0.8 | 0.03 | 3 | First-mover advantage, low entry |
| Commodity | Low differentiation | 0.5 | 0.3 | 0.12 | 12 | Many firms, high turnover, tight margins |

---

## Integration Points

### Backend

- **SessionManager** (`simulation/manager.py`): Creates `MarketEngine` sessions alongside `GrowthEngine` sessions. Mode selection (`"growth"` vs `"market"`) is set at session creation.
- **Routes** (`routes/simulation.py`):
  - `GET /api/simulate/market/presets` — returns preset list
  - `POST /api/simulate/start` with `mode="market"` + `preset` — creates market session
  - SSE stream injects `"mode": "market"` into tick events

### Frontend

- **useMarketSimulation** hook — SSE-connected state for market mode
- **MarketDashboard** — metric cards, stacked area chart, HHI/firms chart, agent table, event log
- **PresetPicker** — 4-card grid for preset selection
- **page.tsx** — mode selector routes to growth or market simulation

---

## Test Coverage

| File | Tests | Coverage Area |
|------|-------|---------------|
| test_market_math.py | 24 | Pure math functions (share, sigmoid, HHI, spawn) |
| test_market_heuristics.py | 18 | All 6 rules, ordering, edge cases |
| test_market_engine.py | 38 | Full tick loop, invariants, redistribution |
| test_market_presets.py | 14 | Preset validity, character, smoke tests |
| test_market_routes.py | 8 | API endpoints, SSE streaming |
| **Total** | **102** | |
