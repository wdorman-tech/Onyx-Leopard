# Market Engine Parameter Reference

## Global Market Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tam_0` | float | 50,000 | Initial total addressable market |
| `g_market` | float | 0.0003 | Per-tick TAM growth rate |
| `lambda_entry` | float | 0.03 | Base new entrant spawn rate per tick |
| `g_ref` | float | 0.001 | Reference growth rate for spawn normalization |
| `alpha` | float | 0.8 | Marketing elasticity in share attraction |
| `beta` | float | 0.8 | Quality elasticity in share attraction |
| `delta` | float | 0.08 | Per-tick churn/decay rate |
| `b_death` | float | -500 | Cash threshold for bankruptcy clock start |
| `t_death` | int | 30 | Consecutive ticks below b_death before death |
| `tau_decay` | int | 60 | Ticks for dead agent's revenue to redistribute |
| `b_threshold` | float | 1,500 | Sigmoid midpoint for capital constraint |
| `k_sigmoid` | float | 0.001 | Sigmoid steepness |
| `t_q` | int | 90 | Ticks between quarterly reviews |
| `c_k` | float | 0.02 | Capacity maintenance cost per unit per tick |
| `capex_per_unit` | float | 0.5 | One-time cost per unit of new capacity |
| `n_0` | int | 5 | Initial number of agents |

## Agent Initialization Ranges

These are used to randomize agent parameters at spawn.

| Parameter | Default Range | Description |
|-----------|--------------|-------------|
| `r_range` | (0.20, 0.35) | Demand capture rate. **Must have r_min > delta.** |
| `margin_range` | (0.20, 0.35) | Gross margin |
| `f_range` | (80, 200) | Fixed costs per tick |
| `q_range` | (0.6, 1.4) | Initial quality score |
| `m_range` | (15, 50) | Initial marketing intensity |
| `k_range` | (800, 2000) | Initial capacity |
| `b_range` | (8000, 20000) | Initial cash balance |
| `eta_m_range` | (0.15, 0.30) | Marketing adjustment speed |
| `eta_q_range` | (0.05, 0.12) | Quality adjustment speed |
| `tau_k_range` | (20, 40) | Capacity expansion delay (ticks) |

---

## Critical Parameter Relationships

### r vs delta (Growth/Churn Balance)

The most important constraint: **r_min must exceed delta** for every preset.

Equilibrium utilization (capacity-bound case):
```
R*/K = r*C / (r*C + delta)
```

| Target Equil. | Required r (C=1) |
|--------------|------------------|
| 50% | r = delta |
| 67% | r = 2 * delta |
| 75% | r = 3 * delta |
| 83% | r = 5 * delta |
| 90% | r = 9 * delta |

Rule of thumb: **r_mid ≈ 3 × delta** gives healthy 75% utilization.

### b_threshold (Sigmoid Sensitivity)

Controls how harshly low cash penalizes growth.

```
C(cash) = 1 / (1 + exp(-k * (cash - b_threshold)))
```

| Cash Level | C value (k=0.001, b_threshold=2000) |
|-----------|-------------------------------------|
| -2,000 | 0.02 (near-zero growth) |
| 0 | 0.12 (heavily penalized) |
| 1,000 | 0.27 (moderate penalty) |
| 2,000 | 0.50 (midpoint) |
| 4,000 | 0.88 (mild penalty) |
| 8,000 | 0.998 (effectively no penalty) |

Setting b_threshold too high creates irreversible death spirals. Setting it too low removes the cash-pressure dynamic entirely. Recommended: b_threshold ≈ 10-20× fixed costs.

### Expansion Rate (Rule 6)

Rule 6 caps expansion at 1.5× per quarter. Growth trajectory:

```
Quarter 1: K → 1.5K
Quarter 2: 1.5K → 2.25K
Quarter 3: 2.25K → 3.375K
...
After N quarters: K * 1.5^N
```

Reaching demand_potential from initial capacity:
- From 1,000 to 20,000: ~8 quarters (~720 days)
- From 1,000 to 10,000: ~6 quarters (~540 days)

### Spawn Rate

Expected time between new entrants (competition channel, C=competitive market):
```
E[ticks] = 1 / (lambda * (1-HHI) * g/g_ref)
```

With demand channel active (90% unserved):
```
E[ticks] = 1 / (lambda * (1-HHI) * g/g_ref + lambda * 0.45)
```

---

## Preset Parameter Values

### Price War

```python
alpha=1.5, beta=0.5, delta=0.15, lambda_entry=0.05, n_0=8
tam_0=80,000, g_market=0.0004
r_range=(0.35, 0.55)
b_threshold=2,000, k_sigmoid=0.0008
b_death=-1,000, t_death=25, tau_decay=45
c_k=0.025, capex_per_unit=0.6
m_range=(30, 80), f_range=(100, 250), b_range=(10k, 25k)
```

### Innovation Race

```python
alpha=0.3, beta=1.5, delta=0.05, lambda_entry=0.03, n_0=5
tam_0=40,000, g_market=0.0005
r_range=(0.12, 0.20)
b_threshold=1,500, k_sigmoid=0.001
b_death=-500, t_death=40, tau_decay=80
c_k=0.015, capex_per_unit=0.4
q_range=(0.5, 1.8), m_range=(10, 30), f_range=(60, 160), b_range=(8k, 18k)
```

### Monopoly

```python
alpha=0.8, beta=0.8, delta=0.03, lambda_entry=0.01, n_0=3
tam_0=60,000, g_market=0.0002
r_range=(0.08, 0.12)
b_threshold=2,000, k_sigmoid=0.001
b_death=-800, t_death=50, tau_decay=100
c_k=0.02, capex_per_unit=0.5
b_range=(15k, 30k), k_range=(1200, 2500)
```

### Commodity

```python
alpha=0.5, beta=0.3, delta=0.12, lambda_entry=0.08, n_0=12
tam_0=100,000, g_market=0.0003
r_range=(0.28, 0.42)
b_threshold=1,000, k_sigmoid=0.0012
b_death=-300, t_death=20, tau_decay=40
c_k=0.03, capex_per_unit=0.3
margin_range=(0.10, 0.25), f_range=(50, 130), m_range=(10, 40)
b_range=(5k, 12k), k_range=(600, 1500)
```
