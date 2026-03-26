# Market Engine Changelog

## 2026-03-23 — Mathematical Overhaul (6 Critical Fixes)

### Summary

Rewrote the core revenue dynamics, expansion heuristics, spawn formula, and death handling to fix 6 structural mathematical flaws that caused all presets to produce dead or stagnant markets.

### Changes by File

#### `backend/src/simulation/market/engine.py`

1. **Step 4 — Revenue equation**: Logistic growth → linear convergence
   - Old: `growth = r * R * (1 - R/M) * C`
   - New: `growth = r * max(0, M - R) * C`
   - Why: Logistic equilibrium R/K = 1 - delta/r was permanently below expansion thresholds

2. **Step 9 — Revenue redistribution**: Dead agents' decaying revenue now redistributed to survivors
   - Proportional to market share, capped by spare capacity
   - Why: Previously, dead competitor revenue just vanished

3. **Step 10 — Spawn probability**: Added demand-driven entry channel
   - New: `P += lambda * unserved_ratio * 0.5`
   - `compute_spawn_probability()` gains optional `unserved_ratio` parameter
   - Why: HHI=1.0 gave P=0 even with 98% unserved TAM

4. **Quarterly review — Thresholds lowered**:
   - Rule 3 (Capacity Constraint): 0.85 → 0.75
   - Rule 5 (Profitable Growth): 0.70 → 0.60
   - Why: These were unreachable with the churn-adjusted equilibrium

5. **Quarterly review — Rule 6 added** (Market Opportunity):
   - Triggers when demand_potential > capacity × 1.2 and cash > 3×F
   - Expands toward demand, capped at 1.5× per quarter
   - Method signature changed: `_quarterly_review(agent)` → `_quarterly_review(agent, tam)`
   - Why: Agents never expanded despite massive unserved demand

#### `backend/src/simulation/market/models.py`

- `r_range` default: (0.04, 0.08) → (0.20, 0.35)
- `b_threshold` default: 5,000 → 1,500

#### `backend/src/simulation/market/presets.py`

All 4 presets updated:

| Preset | r_range (new) | b_threshold (new) |
|--------|--------------|-------------------|
| Price War | (0.35, 0.55) | 2,000 |
| Innovation Race | (0.12, 0.20) | 1,500 |
| Monopoly | (0.08, 0.12) | 2,000 |
| Commodity | (0.28, 0.42) | 1,000 |

#### `backend/tests/test_market_heuristics.py`

- All `_quarterly_review()` calls updated to pass `tam` parameter
- Added 4 Rule 6 tests (expansion, low demand, broke, cap)
- Added Rule 6 ordering test
- Adjusted test values for lowered thresholds
- Agent r values updated to match new ranges

#### `backend/tests/test_market_engine.py`

- Added `TestRevenueDynamics` class with linear convergence equilibrium test
- Added `TestRevenueRedistribution` class (redistribution to survivors, capacity cap)
- Added `TestNewEntrants::test_entrants_spawn_with_unserved_demand`
- Added `TestCapacityExpansion::test_market_opportunity_drives_expansion`
- Added `TestGrowthChurnBalance` class (r > delta validation, equilibrium > 50%)

#### `backend/tests/test_market_math.py`

- Spawn probability tests updated for dual-channel formula
- Added: `test_monopoly_hhi_nonzero_with_unserved_demand`
- Added: `test_unserved_demand_boosts_entry`
- Added: `test_unserved_ratio_clamped`

#### `backend/tests/test_market_presets.py`

- Added: `test_all_presets_r_exceeds_delta`

### Test Results

- **110 tests, 110 passed, 0 failed**
- **800-tick stress test**: All 4 presets stable, 64-74% TAM captured, dynamic entry/exit

### Before/After

| Metric | Before (Monopoly, tick 600) | After (Monopoly, tick 800) |
|--------|---------------------------|---------------------------|
| Alive agents | 1 of 3 | 6 of 6 |
| TAM captured | ~2% | ~64% |
| HHI | 1.0 (monopoly) | 0.175 (moderate) |
| Capacity expansion | Never | Active |
| New entrants | Never (P=0) | Yes (3 spawned) |
| Revenue trend | Flat/declining | Growing with TAM |
