# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Onyx Leopard — Guiding Principles

Every build decision, feature addition, and refactor must align with these principles. This is the constitution of the project. Read it before writing code.

## What This Is

Onyx Leopard is an open-source, local-run business simulation framework. It models any business as a node-based growth system where AI agents operate departments, make decisions, and react to market conditions. The simulation produces emergent outcomes — not scripted results.

**The end goal:** Run 10+ parallel simulations of slightly different business configurations simultaneously, Monte Carlo style, to determine optimal structure, strategy, and resilience. Stress-test a business against market shocks, competitive pressure, and internal failures before it exists in the real world.

## Common Commands

**One-shot setup + launch (the supported entry point):**
```bash
python start.py    # checks Python 3.12+/Node/pnpm, installs deps, prompts for ANTHROPIC_API_KEY,
                   # frees ports 8000/3000, then runs uvicorn + pnpm dev together
```

**Backend (run from `backend/`):**
```bash
uv sync --group dev                                # install (or: pip install -e . for the runtime only)
uv run uvicorn src.main:app --reload --port 8000   # run API alone
uv run pytest -v                                   # full suite
uv run pytest -v -m "not slow"                     # skip multi-year/long-running tests
uv run pytest tests/test_unified_v2.py::test_name  # single test
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

The pytest `slow` marker is registered in `backend/pyproject.toml`; respect it when adding long sims. Tests rely on `src.*` imports — always invoke from `backend/`, not the repo root.

**Frontend (run from `frontend/`):**
```bash
pnpm install
pnpm dev          # next dev on port 3000
pnpm build        # production build (use to catch type errors)
pnpm lint         # eslint
```

There is no frontend test runner configured. Type-check via `pnpm build` or `tsc --noEmit`.

**Configuration:** `backend/.env` holds `ANTHROPIC_API_KEY` (template at `.env.example`). Without it the sim runs in pure-math mode (no LLM orchestrator calls).

## Architecture Quick Reference

Concrete pointers so future instances don't have to crawl the tree to find the load-bearing files:

- **API entry:** `backend/src/main.py` mounts `simulation_v2_router` at `/api/v2/simulate` with SSE streaming and CORS for `localhost:3000`. Routes live in `backend/src/routes/simulation_v2.py`.
- **V2 simulation engine:** `backend/src/simulation/unified_v2.py` — `MultiCompanySimV2` runs N `CompanyAgentV2` instances per tick. This is the active engine. The legacy v1 `unified.py` / `IndustrySpec` path is being deleted (wave 5).
- **Inputs to the v2 engine:** `seed.py` (`CompanySeed`, ~31 fields, archetype samplers), `stance.py` (`CeoStance`, locked CEO persona), `library_loader.py` (`NodeLibrary`, loads `node_library.yaml`), `shocks.py` (`ShockScheduler`).
- **Orchestrator triad** (`backend/src/simulation/orchestrator.py`): `HeuristicOrchestrator` (Tier 1, free) → `TacticalOrchestrator` (Haiku) → `StrategicOrchestrator` (Sonnet). The `OrchestratorBundle` decides which nodes spawn/retire each tick. `critic.py` validates strategic decisions.
- **Pure math layer** lives at **`backend/src/biosim/`** (NOT a separate top-level `src/biosim/` — older docs and the README still say that): `math/production.py` (Cobb-Douglas), `math/competition.py` (Lotka-Volterra), `math/solver.py` (RK45 → BDF → Euler), `math/growth.py`, `engine/tick.py`, `agents/mesa_model.py`. v2 imports these directly (`from src.biosim.math.competition import ...`).
- **Standalone market engine:** `backend/src/simulation/market/engine.py` (`MarketEngine`) with presets in `presets.py` — the price-war / innovation-race / monopoly / commodity scenarios used by the market dashboard.
- **Bridge derivation:** `bridge.py` derives modifier aggregates each tick from the live spawned-node set. There is no static modifier-key declaration in v2 (intentional — see the file header on the empty-modifier-key bug class it eliminates).
- **Frontend:** Next.js App Router under `frontend/src/app/`, force-directed canvas graph at `frontend/src/components/graph/ForceGraph.tsx` (the visual reference — see "Visualization" below), v2 SSE client in `frontend/src/lib/apiV2.ts` and `frontend/src/hooks/useV2Simulation.ts`.

**Frontend caveat:** `frontend/AGENTS.md` warns that Next.js 16 has breaking changes from prior versions — App Router conventions in your training data may be wrong. Read `node_modules/next/dist/docs/` before writing Next.js code, and heed deprecation notices.

**Path drift to watch for:** The README's "Project Structure" and Development sections still describe `src/biosim/` and a top-level `tests/` — those are stale. Real locations are `backend/src/biosim/` and `backend/tests/`.

## Core Vision

- **Universal business modeling.** Any industry, any company archetype, any scale — from a solo founder to a Fortune 500. The engine is the same; only the data (YAML configs) changes.
- **Emergence over scripting.** Outcomes are never hardcoded or predetermined. Nodes interact through defined channels, and interesting dynamics surface naturally. If you can predict every outcome before running the sim, the sim is too simple.
- **Monte Carlo optimization.** The platform should support running many slightly-varied simulations in parallel and comparing results statistically to find optimal configurations.
- **Stress testing.** Inject market crashes, competitor entries, supply chain disruptions, regulatory changes, key employee departures — and watch how the business responds.

## Design Principles

### 1. Nothing Is Hardcoded

If a value, behavior, or structure is specific to one industry or company type, it belongs in a YAML config — not in the engine. The engine must be completely industry-agnostic. A restaurant, a SaaS startup, and a manufacturing plant should all run through the same code paths with different data.

**Test:** If you deleted every YAML config, would the engine still compile and run (with no data)? It should.

### 2. Robust Over Easy

Never take the shortcut. If a calculation can be done with a proper mathematical model (ODE, logistic growth, Lotka-Volterra competition), use the model. Don't approximate with `if/else` chains. Don't use magic numbers. Don't fake emergence with random noise.

The framework exists to produce trustworthy results. Cutting corners in the math makes the entire tool worthless.

### 3. Plug and Play

Adding a new industry should require:
1. Writing a YAML config file
2. Optionally writing industry research docs
3. Nothing else

No engine changes. No new Python files. No frontend modifications. If adding an industry requires touching engine code, the abstraction is wrong.

### 4. Emergent Behavior From Agent Interaction

Inspired by MiroFish's swarm intelligence architecture: nodes/agents should influence each other through defined interaction channels. A Sales node revenue spike should propagate back-pressure into Ops and Hiring. A supply chain disruption should cascade through dependent nodes. The P&L and company health should emerge from these interactions, not from top-down formulas.

### 5. Persistent Node State

Each node/agent carries its own running state history. Behavior in tick N+1 is shaped by what happened in ticks N-5 through N, not just the current snapshot. Departments have memory — a department that was starved of budget for 10 ticks should behave differently than one that was well-funded.

### 6. Separation of Concerns

Three distinct layers, never mixed:
- **Pure math layer** (`backend/src/biosim/`): ODEs, growth models, competition models via Mesa/NumPy/SciPy. No business logic, no API calls, no UI concerns.
- **Business logic layer** (`backend/src/simulation/` + `backend/src/routes/`): FastAPI routes, session management, seed/stance/library inputs, orchestrator triad, SSE streaming. Active engine is `MultiCompanySimV2` in `unified_v2.py`; `MarketEngine` is the standalone market-dashboard engine. The legacy v1 `UnifiedEngine` path is being retired. Shared math is consolidating into the pure math layer — v2 already imports `production`, `competition`, and `solver` directly from `src.biosim`.
- **Presentation layer** (`frontend/src/`): Visualization, controls, dashboards. Consumes SSE streams. No simulation logic.

### 7. Data-Driven Configuration

Follow the pattern established by the YAML industry configs:
- Node taxonomies, triggers, growth stages, bridge parameters, location economics — all declared in data
- The engine reads the data and executes generically
- New behaviors come from new data patterns, not new code paths

### 8. Multi-Simulation Architecture

Design every feature with the assumption that 10 instances will run simultaneously:
- State must be isolated per simulation instance
- No global mutable state
- Memory and CPU usage must scale linearly, not exponentially
- Results must be comparable across runs (consistent output schema)

## Mathematical Rigor

The simulation's value comes from its mathematical foundation. These are non-negotiable:

- **Production:** Cobb-Douglas model (`Y = A * K^α * L^β`) — vectorized across all agents
- **Growth:** Logistic growth ODE with RK45 integration (SciPy), BDF fallback for stiff systems, Euler as last resort
- **Competition:** Lotka-Volterra N-species model with competition matrices
- **Market share:** Multinomial logit (`s_i ∝ q_i^β * m_i^α`)
- **Revenue dynamics:** Linear convergence with sigmoid capital constraint and churn
- **Spawn/death:** Dual-channel probabilistic entry, multi-tick insolvency death

When adding new mechanics, find the established mathematical model for that phenomenon. Don't invent ad-hoc formulas.

## AI Agent Architecture

### Current State
- v2 uses an orchestrator triad (`backend/src/simulation/orchestrator.py`): `HeuristicOrchestrator` (rule-based, every tick, free) → `TacticalOrchestrator` (Haiku) → `StrategicOrchestrator` (Sonnet). The `CriticAgent` (`critic.py`) scores strategic proposals.
- CEO persona is locked at start via `CeoStance` (`stance.py`); company shape comes from `CompanySeed` (`seed.py`) — both are inputs to `/api/v2/simulate/start`.
- The orchestrator decides which nodes spawn/retire from a universal `NodeLibrary` (`node_library.yaml`); bridge modifiers are derived each tick from the live spawned set, not declared statically.
- `CostTracker` (`replay.py`) enforces a per-run dollar ceiling; `Transcript` records the full decision stream for replay.
- All simulation math runs without AI — AI is an overlay, not a dependency. With no `ANTHROPIC_API_KEY` the heuristic tier handles every decision.

### Target Architecture
- Agents operate within the simulation tick loop, not called externally by routes
- Tiered AI: pure math (most computation) → heuristic rules (quarterly) → Haiku (operational agents) → Sonnet (executive agents)
- Per-agent persistent memory that shapes behavior across the full simulation run
- Probabilistic activation — agents fire based on domain cadence, not fixed intervals
- Role-locked via inception prompting so agent personas don't drift
- Budget-tracked: every AI call has a cost ceiling per simulation run

## Inspiration: MiroFish + CAMEL-AI Patterns

This project draws from two complementary open-source systems:

- **MiroFish** — a multi-agent swarm intelligence prediction engine that turns seed material into a simulated world of autonomous agents producing emergent outcomes.
- **CAMEL-AI** — the multi-agent framework MiroFish is built on. Its OASIS engine scales to 1M+ agents with persistent memory, role-locked behavior, and hierarchical coordination.

### From MiroFish

1. **Seed-to-world pipeline:** A company profile (seed) feeds a knowledge graph of departments and relationships, which populates node parameters and initial state — rather than requiring manual configuration of everything.
2. **Per-agent memory:** Every agent carries long-term memory that shapes future behavior. Departments remember their history.
3. **Parallel environment simulation:** Running the same company under different market conditions simultaneously and diffing the results.
4. **Synthesis agent:** A dedicated agent that sits outside the simulation, watches all nodes, and produces structured insights — separate from the simulation mechanics.
5. **Graph-first representation:** Relationships between entities are the primary data structure. The graph IS the simulation state.

### From CAMEL-AI

6. **Environment server pattern:** A single authoritative state store (not agent memory) is the source of truth. Agents read from it and write back through defined actions. No agent holds authoritative state in its own memory.
7. **Probabilistic activation — not every agent fires every tick.** Sales makes decisions when deals close. Marketing follows campaign cycles. Budget decisions are quarterly. Model activation frequencies domain-specifically rather than running all agents every tick.
8. **Inception prompting for role consistency.** When spawning autonomous agents (CEO, CFO, Sales VP), their roles must be locked via explicit system prompt constraints. A cautious CFO agent must never drift into aggressive growth marketer reasoning.
9. **Workforce coordinator pattern.** High-level "company strategy" decomposes into department-level quarterly objectives, then into agent-level actions. The coordinator routes tasks by natural language description of worker capability — no hardcoded routing.
10. **Scale-dependent emergence.** Small simulations can lie. A simulation with 3 customer agents will not show the same market dynamics as one with 300. Population size matters for result validity.
11. **Heterogeneous agent tiers.** Use cheaper/faster models for lower-level operational agents and more capable models for executive-level strategic decisions. Match model capability to role complexity.
12. **CriticAgent for decision validation.** Agents evaluating strategic proposals generate multiple candidate decisions, score them, and select the best — with explicit reasoning attached to the choice.

## Visualization — The Graph Is The Product

The force-directed node graph is the defining visual of Onyx Leopard. The current implementation in `frontend/src/components/graph/ForceGraph.tsx` is the reference for what the final system should look like. Do not replace, redesign, or deviate from this visualization style.

### Visual Spec (preserve exactly)

- **Canvas-rendered d3-force simulation** — not SVG, not React Flow, not a library graph component. Raw canvas with `requestAnimationFrame` loop for performance at scale.
- **Force layout:** d3-force with charge repulsion, link attraction, center gravity, collision detection, radial layout with fixed root node (single company), and custom clustering force with free-floating roots (multi-company).
- **Node categories with distinct colors:**
  - Root: `#ec4899` (pink), radius 14
  - Location: `#22c55e` (green), radius 8
  - Revenue: `#8b5cf6` (purple), radius 7
  - Corporate: `#3b82f6` (blue), radius 6
  - External: `#f59e0b` (amber), radius 5
- **Glow halos** around every node with category-color tint (subtle transparency).
- **Multi-company mode:** Separate company clusters with strong root repulsion (-600 charge), custom clustering force pulling child nodes toward their company root. Company name labels under root nodes.
- **Dead companies** rendered with muted slate color (`#94a3b8`) and reduced opacity.
- **Minimap** in bottom-right corner showing the full graph with a viewport rectangle.
- **Pan and zoom** via mouse drag and scroll wheel.
- **Click-to-select** nodes with tooltip overlay showing node label, type, category, and metrics.
- **Hover effects:** node turns dark, gets a colored ring, shows label text.
- **White background**, subtle gray edges (`rgba(148, 163, 184, 0.35)`).

### Why This Matters

The graph is not a debug view or a developer tool — it IS the product. Users watch their business grow node by node. The force-directed layout makes organic growth patterns visible. Company clusters naturally separate. Departments orbit their root. The visualization communicates the simulation state at a glance without needing to read numbers.

Any new feature that adds nodes, edges, or visual state must integrate into this existing graph system. Do not create separate visualization components for new data — extend the graph.

## What Not To Do

- **Don't hardcode industry-specific logic in the engine.** Ever.
- **Don't use magic numbers.** Every constant must be named, documented, and configurable.
- **Don't fake emergence.** If the only way to produce an interesting outcome is `random.random() > 0.7`, the model is wrong.
- **Don't optimize prematurely.** Correctness first, then performance. But design for 10 parallel sims from the start.
- **Don't add features that only work for one industry.** If it can't generalize, it doesn't belong in the engine.
- **Don't skip the math.** If there's an established model for a phenomenon, use it. Don't approximate with conditionals.
- **Don't break the plug-and-play contract.** Adding a new industry = adding a YAML file. Period.

## Tech Stack

- **Backend:** Python >=3.12, FastAPI, NumPy 2, SciPy, Pydantic 2, camel-ai (multi-agent framework), edgartools (SEC EDGAR)
- **Pure math layer:** Mesa 3, NumPy, SciPy, NetworkX — at `backend/src/biosim/` (a sub-package of the backend, not a separate top-level project)
- **Frontend:** Next.js 16, React 19, d3-force (canvas-rendered), Recharts, TailwindCSS 4, TypeScript (strict). Note: Next.js 16 has breaking changes — see `frontend/AGENTS.md`.
- **AI:** Anthropic Claude API — Sonnet for the strategic orchestrator + critic, Haiku for the tactical orchestrator. Heuristic tier needs no key.
- **Configuration:** Universal `node_library.yaml` (no per-industry specs in v2), JSON company profiles (.oleo format), `CompanySeed` + `CeoStance` Pydantic models on the API.
- **Deployment:** Local only. No cloud services. `start.py` is the single entry point.

## Project Structure

```
backend/src/main.py              # FastAPI app, CORS, /health, mounts simulation_v2_router
backend/src/routes/              # FastAPI API routes (simulation_v2.py — SSE streaming)
backend/src/simulation/          # Business logic: unified_v2, orchestrator, seed, stance,
                                 #   library_loader, bridge, shocks, critic, replay, monte_carlo_v2
backend/src/simulation/market/   # Standalone MarketEngine + presets (separate from v2 unified engine)
backend/src/simulation/node_library.yaml  # Universal node taxonomy (replaces v1 industry specs)
backend/src/biosim/              # Pure math engine (NumPy/SciPy/Mesa) — math/, engine/, agents/, types/
backend/tests/                   # pytest suites (test_unified_v2, test_growth_invariants,
                                 #   test_orchestrator_*, test_market_*, test_biosim_integration, ...)
frontend/src/                    # Next.js frontend (App Router)
frontend/src/components/graph/ForceGraph.tsx  # The canvas force graph — visual reference
frontend/AGENTS.md               # READ FIRST when editing frontend — Next.js 16 caveats
documentation/                   # Architecture docs (market_engine_*)
research/industries/             # Industry research (sourced) — 15 industries
start.py                         # Single entry point: deps + dual-server launch
.env.example                     # Template; real key lives in backend/.env
```
