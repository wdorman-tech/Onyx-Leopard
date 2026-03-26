# Competitive Market Simulation: Mathematical Specification

**v0.1 MVP**

Agent-Based Logistic Growth Model with Competitive Share Attraction and Dynamic Entry/Exit

Target Runtime: Mesa Framework (Python) | March 2026

---

## 1. Overview

This document specifies a generic, domain-agnostic agent-based model for simulating competitive market dynamics. The model combines logistic growth mechanics with a multinomial share attraction function, enabling emergent behaviors such as market concentration, price wars, and natural monopoly formation.

The model is designed for integration into a Mesa-based simulation engine with a PyQt/pyqtgraph visualization layer. Agent decisions are driven by heuristic rules (if/then logic) at two temporal scales. No exogenous shock events are included in the MVP scope.

---

## 2. State Variables

### 2.1 Per-Agent State

Each agent `i` maintains the following state vector at discrete time step `t`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `R_i(t)` | float >= 0 | Revenue: current captured demand in currency units per tick |
| `B_i(t)` | float | Cash balance: liquid capital available for operations and investment |
| `K_i(t)` | float > 0 | Capacity: maximum revenue the agent can service given current infrastructure |
| `q_i(t)` | float > 0 | Quality score: abstract measure of product/service competitiveness |
| `m_i(t)` | float >= 0 | Marketing intensity: current spend on demand generation |
| `q_target_i` | float > 0 | Target quality: strategic goal set at quarterly review |
| `m_target_i` | float >= 0 | Target marketing: strategic goal set at quarterly review |
| `K_target_i` | float > 0 | Target capacity: expansion goal committed at quarterly review |
| `alive_i` | bool | Whether the agent is active in the simulation |

### 2.2 Global State

The model object maintains:

| Parameter | Type | Description |
|-----------|------|-------------|
| `TAM(t)` | float > 0 | Total addressable market: aggregate demand available to all agents |
| `N(t)` | int >= 0 | Number of active (alive) agents |
| `t` | int >= 0 | Current simulation tick |
| `T_q` | int > 0 | Ticks per quarterly strategic review cycle |

---

## 3. Core Equations

### 3.1 TAM Evolution

The total addressable market grows (or shrinks) at a constant exogenous rate:

```
TAM(t+1) = TAM(t) * (1 + g_market)
```

`g_market` is the per-tick market growth rate. Positive values model expanding markets, negative values model contraction. For the MVP, this is a constant. Future versions may support stochastic or shock-driven TAM dynamics.

### 3.2 Share Attraction (Multinomial Logit)

Each agent's share of market attention is computed via a gravity model:

```
s_i(t) = (q_i(t)^beta * m_i(t)^alpha) / SUM_j(q_j(t)^beta * m_j(t)^alpha)
```

The sum runs over all alive agents `j`. The elasticity parameters control the market's structural character:

- `alpha > beta`: Marketing-driven market. Spend determines share. Favors well-capitalized incumbents.
- `beta > alpha`: Product-driven market. Quality determines share. Favors innovative entrants.
- `alpha = beta`: Balanced market. Both dimensions matter equally.

*Edge case: When `m_i(t) = 0` and `alpha > 0`, the agent's share attraction is zero regardless of quality. This is intentional: a firm with zero marketing has zero market visibility. If this is undesirable, add a floor: `m_i(t) = max(m_i(t), epsilon)`.*

### 3.3 Effective Market Ceiling

Each agent's growth is bounded by the minimum of their demand allocation and supply capacity:

```
M_i(t) = min(TAM(t) * s_i(t), K_i(t))
```

This captures the fundamental constraint: you cannot grow faster than either your market share allows or your infrastructure supports. The binding constraint switches dynamically as agents invest in capacity or gain/lose share.

### 3.4 Revenue Dynamics (Logistic Growth with Churn)

Revenue evolves according to a discrete logistic growth equation with a decay term:

```
R_i(t+1) = R_i(t) + r_i * R_i(t) * (1 - R_i(t)/M_i(t)) * C_i(t) - delta * R_i(t)
```

Components:

- `r_i * R_i(t) * (1 - R_i(t)/M_i(t))` is the logistic growth term. Growth accelerates when R is small relative to M, decelerates as R approaches M, and halts at saturation.
- `C_i(t)` is the capital constraint multiplier (see Section 3.5). Scales growth based on cash health.
- `delta * R_i(t)` is the churn/decay term. A constant fraction of revenue is lost each tick. This is essential: without churn, agents cannot lose market share to competitors. Churned revenue returns to the contestable pool implicitly through reduced `R_i`.

**Stability constraint:** To prevent negative revenue, enforce `R_i(t+1) = max(R_i(t+1), 0)` after the update.

### 3.5 Capital Constraint

Cash health maps to a smooth multiplier on growth:

```
C_i(t) = 1 / (1 + exp(-k * (B_i(t) - B_threshold)))
```

This is a standard sigmoid. When `B_i >> B_threshold`, C approaches 1 (no drag). When `B_i << B_threshold`, C approaches 0 (growth stalls). The parameter `k` controls the steepness of the transition. Higher `k` means a sharper cliff between healthy and unhealthy cash positions.

### 3.6 Cash Balance Dynamics

```
B_i(t+1) = B_i(t) + R_i(t) * margin_i - c_K * K_i(t) - m_i(t) - F_i
```

Where:

- `margin_i`: Gross margin. Fraction of revenue retained after cost of goods/services.
- `c_K * K_i(t)`: Capacity maintenance cost. Proportional to installed capacity, not utilized capacity. Idle capacity still burns cash.
- `m_i(t)`: Marketing spend (direct outflow).
- `F_i`: Fixed operating costs (rent, payroll, overhead).

---

## 4. Two-Layer Decision Model

### 4.1 Continuous Adjustments (Every Tick)

Between quarterly reviews, agents smoothly drift toward their strategic targets:

```
m_i(t+1) = m_i(t) + eta_m * (m_target_i - m_i(t))
q_i(t+1) = q_i(t) + eta_q * (q_target_i - q_i(t))
```

The `eta` parameters are adjustment speeds in (0, 1]. `eta_m` should typically be higher than `eta_q` because marketing spend can be changed quickly while quality improvements (R&D, process improvement) are inherently slow.

### 4.2 Quarterly Strategic Review (Every T_q Ticks)

At quarterly boundaries (when `t mod T_q = 0`), each alive agent executes a heuristic decision function to update its strategic targets. The heuristic is a rule-based function of the agent's current state and observable market conditions.

#### 4.2.1 Heuristic Decision Rules (MVP)

The following rules execute in order. Each rule sets one or more target variables. Later rules can override earlier ones.

**Rule 1: Cash Emergency**
```
IF B_i(t) < 2 * F_i THEN:
    m_target_i = m_i(t) * 0.5      (cut marketing)
    K_target_i = K_i(t)             (freeze capacity)
```

**Rule 2: Excess Capacity**
```
IF R_i(t) / K_i(t) < 0.5 THEN:
    m_target_i = m_i(t) * 1.2       (increase marketing to fill capacity)
```

**Rule 3: Capacity Constraint**
```
IF R_i(t) / K_i(t) > 0.85 THEN:
    K_target_i = K_i(t) * 1.3       (expand capacity by 30%)
```

**Rule 4: Market Share Decline**
```
IF s_i(t) < s_i(t - T_q) * 0.9 THEN:
    q_target_i = q_i(t) * 1.1       (invest in quality)
    m_target_i = m_i(t) * 1.15      (boost marketing)
```

**Rule 5: Profitable Growth**
```
IF B_i(t) > 5 * F_i AND R_i(t)/K_i(t) > 0.7 THEN:
    q_target_i = q_i(t) * 1.05      (incremental quality)
    K_target_i = K_i(t) * 1.15      (moderate expansion)
```

*These multipliers are tunable per-agent or per-preset. The rule structure is fixed for MVP. Future versions may replace this with LLM-routed decisions for complex strategic reasoning.*

### 4.3 Capacity Expansion Delay

Capacity changes are not instant. When `K_target_i > K_i(t)` is set at a quarterly review, the actual capacity update occurs after `tau_K` ticks:

```
K_i(t + tau_K) = K_target_i
```

During the build period, the agent pays a one-time capital expenditure proportional to the capacity increment. This is deducted from the cash balance at the time of commitment, not delivery.

---

## 5. Agent Entry and Exit

### 5.1 Exit (Bankruptcy)

An agent is marked dead when its cash balance falls below a death threshold for a sustained period:

```
IF B_i(t) < B_death FOR T_death consecutive ticks THEN alive_i = false
```

On death:

- The agent is removed from the share attraction denominator immediately.
- The agent's captured revenue `R_i` decays back into the contestable pool over `tau_decay` ticks. This models the lag in customer reallocation: customers of a dead firm don't instantly find alternatives.

### 5.2 Entry (New Entrants)

New agents spawn probabilistically based on market attractiveness:

```
P(spawn | t) = lambda * max(0, 1 - HHI(t)) * max(0, g_market) / g_ref
```

Where:

- `HHI(t)` is the Herfindahl-Hirschman Index: `SUM_i(s_i^2)`. Ranges from `1/N` (perfect competition) to `1` (monopoly). Higher concentration discourages entry.
- `lambda` is the base entry rate per tick.
- `g_ref` is a reference growth rate for normalization.

**New entrant initialization:**

- `R = 0` (no initial customers)
- `B = B_0_entrant` (startup capital, drawn from a configurable distribution)
- `K = K_0_entrant` (small initial capacity)
- `q, m` drawn from uniform distributions over configurable ranges

---

## 6. Full Parameter Reference

### 6.1 Market Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `TAM_0` | float > 0 | Initial total addressable market |
| `g_market` | float | Per-tick market growth rate (can be negative) |
| `lambda` | float >= 0 | Base new entrant spawn rate per tick |
| `g_ref` | float > 0 | Reference growth rate for entry normalization |
| `B_death` | float | Cash threshold below which bankruptcy clock starts |
| `T_death` | int > 0 | Consecutive ticks below B_death before death |
| `tau_decay` | int > 0 | Ticks for dead agent's customers to reallocate |

### 6.2 Competition Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | float >= 0 | Marketing elasticity in share attraction function |
| `beta` | float >= 0 | Quality elasticity in share attraction function |
| `delta` | float in [0,1] | Per-tick churn/decay rate on revenue |

### 6.3 Agent Parameters

These can be set per-agent or drawn from distributions:

| Parameter | Type | Description |
|-----------|------|-------------|
| `r_i` | float > 0 | Intrinsic growth rate (product-market fit, sales efficiency) |
| `margin_i` | float in (0,1) | Gross margin (fraction of revenue retained) |
| `F_i` | float >= 0 | Fixed operating costs per tick |
| `c_K` | float >= 0 | Capacity maintenance cost per unit of installed capacity per tick |
| `eta_m` | float in (0,1] | Marketing adjustment speed (continuous) |
| `eta_q` | float in (0,1] | Quality adjustment speed (continuous) |
| `tau_K` | int >= 0 | Capacity build delay in ticks |
| `B_0` | float | Starting cash balance |
| `K_0` | float > 0 | Starting capacity |
| `q_0` | float > 0 | Starting quality score |
| `m_0` | float >= 0 | Starting marketing intensity |

### 6.4 Meta Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `T_q` | int > 0 | Ticks per quarterly strategic review cycle |
| `B_threshold` | float | Cash level at sigmoid midpoint for capital constraint |
| `k` | float > 0 | Steepness of capital constraint sigmoid |
| `N_0` | int > 0 | Initial number of agents |

---

## 7. Preset Parameter Bundles

Presets provide semantically meaningful starting configurations. Users can load a preset and then override individual parameters.

| Preset | alpha | beta | delta | lambda | N_0 | Character |
|--------|-------|------|-------|--------|-----|-----------|
| Price War | 1.5 | 0.5 | 0.15 | 0.05 | 8 | High marketing spend drives share; high churn |
| Innovation Race | 0.3 | 1.5 | 0.05 | 0.03 | 5 | Quality determines winners; slow churn |
| Monopoly | 0.8 | 0.8 | 0.03 | 0.01 | 3 | Low entry, low churn; first mover wins |
| Commodity | 0.5 | 0.3 | 0.12 | 0.08 | 12 | Many substitutable firms; high turnover |

---

## 8. Simulation Loop (Pseudocode)

The following pseudocode defines the per-tick execution order:

```
FOR each tick t:

    1. UPDATE TAM
       TAM(t+1) = TAM(t) * (1 + g_market)

    2. COMPUTE SHARE ATTRACTION
       FOR each alive agent i:
           s_i = (q_i^beta * m_i^alpha) / SUM_j(q_j^beta * m_j^alpha)

    3. COMPUTE EFFECTIVE CEILING
       FOR each alive agent i:
           M_i = min(TAM * s_i, K_i)

    4. UPDATE REVENUE
       FOR each alive agent i:
           C_i = sigmoid(B_i, B_threshold, k)
           R_i += r_i * R_i * (1 - R_i/M_i) * C_i - delta * R_i
           R_i = max(R_i, 0)

    5. UPDATE CASH
       FOR each alive agent i:
           B_i += R_i * margin_i - c_K * K_i - m_i - F_i

    6. CONTINUOUS ADJUSTMENT
       FOR each alive agent i:
           m_i += eta_m * (m_target_i - m_i)
           q_i += eta_q * (q_target_i - q_i)

    7. QUARTERLY REVIEW (if t mod T_q == 0)
       FOR each alive agent i:
           Execute heuristic rules (Section 4.2.1)
           Commit capacity expansion if K_target > K

    8. CAPACITY DELIVERY
       FOR each pending expansion reaching tau_K:
           K_i = K_target_i

    9. DEATH CHECK
       FOR each alive agent i:
           IF B_i < B_death: increment death counter
           ELSE: reset death counter
           IF death counter >= T_death: kill agent

   10. SPAWN CHECK
       Compute HHI = SUM_i(s_i^2)
       P_spawn = lambda * max(0, 1-HHI) * max(0, g_market) / g_ref
       IF random() < P_spawn: create new agent
```

---

## 9. Key Observables for Visualization

The following metrics should be computed and exposed to the pyqtgraph visualization layer each tick:

- Total market captured: `SUM_i(R_i)` and ratio to TAM
- Market share distribution: `s_i` for all alive agents (stacked area chart)
- HHI over time (market concentration trend)
- Agent count `N(t)` over time
- Per-agent revenue, cash, capacity trajectories
- Capacity utilization: `R_i / K_i` per agent
- Binding constraint indicator: whether demand or capacity is the ceiling per agent

---

## 10. Known Limitations (MVP)

- No exogenous shocks (TAM shocks, regulation, technology disruption, seasonality)
- No multi-segment markets or cross-entry dynamics
- Heuristic agent decisions only (no LLM-routed strategic reasoning)
- No network effects or demand-side economies of scale
- No geographic or spatial dimension
- Capacity expansion is a single step function, not incremental construction
- No debt/financing mechanics (agents cannot borrow)
- Share attraction uses a static functional form (no learning or adaptation of alpha/beta)
