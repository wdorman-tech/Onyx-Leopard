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
python start.py    # checks Python 3.12+/Node/pnpm, prompts for + verifies OPENROUTER_API_KEY,
                   # installs deps (uv if present), frees ports 8000/3000, runs uvicorn + pnpm dev
```

**Backend (run from `backend/`):**
```bash
uv sync --group dev                                # install (or: pip install -e . for the runtime only)
uv run uvicorn src.main:app --reload --port 8000   # run API alone
uv run pytest -v                                   # full suite
uv run pytest -v -m "not slow"                     # skip multi-year/long-running tests
uv run pytest tests/test_engine.py::test_name     # single test
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

**Configuration:** `backend/.env` holds `OPENROUTER_API_KEY` (template at `.env.example`). Without it the sim runs with heuristic CEOs only (no LLM calls); the onboarding screen can save one via `POST /api/config/openrouter-key` (verifies with OpenRouter, never overwrites an existing key, never echoes it back). `OPENROUTER_BASE_URL` optionally points the client at a proxy or local stand-in.

## Architecture Quick Reference

Concrete pointers so future instances don't have to crawl the tree to find the load-bearing files:

- **API entry:** `backend/src/main.py` — FastAPI app with CORS for `localhost:3000`. Routes: `/api/config`, `/api/models`, `/api/archetypes`, `/api/companies`, `/api/seed/sample/{archetype}`, `/api/simulate/start|stream/{id}|control/{id}` (SSE stream).
- **Engine:** `backend/src/simulation/engine.py` — `MarketSim` ticks N `Company` instances (one tick = one day) in a shared market. All companies in a run share one `CompanySeed`; they differ by CEO stance.
- **Inputs:** `seed.py` (`CompanySeed`, archetype sampler that derives a coherent seed from the library, `.oleo` profile conversion), `stance.py` (`CeoStance`, locked persona + archetype profiles), `library.py` (`NodeLibrary` from `node_library.yaml`, plus the suffix-based modifier buckets), `shocks.py` (table-driven shock types; market-scope shocks hit every company, company-scope shocks one).
- **CEOs** (`backend/src/simulation/ceo.py`): `HeuristicCeo` (free, weekly; `propose` → `commit`/`defer`) and `AiCeo` (OpenRouter, monthly + on severe shocks). `sanitize()` enforces prerequisites, caps and engine bounds on every decision. `report.py` writes end-of-run post-mortems.
- **Jev** (`backend/src/simulation/jev.py`): optional TypeSafe System One judge for the weekly heuristic proposals only. One Noul per proposal, all numbers precomputed, gate on a rolling per-company baseline (EMA), not 0.5. Needs `TYPESAFE_API_KEY`.
- **Run record + report:** `Company.series` / `MarketSim.market_series` hold the whole-run time series (≤ `MAX_SERIES_POINTS` samples); `export.py` saves each finished run to `backend/runs/<run_id>/` (`run.json`, `timeseries.csv`, `report.pdf`) and `report_pdf.py` renders the in-theme PDF from `run.json` alone. Served at `/api/runs/{run_id}/{file}`.
- **OpenRouter client** (`backend/src/simulation/llm.py`): strict JSON-schema output routed to providers that support it (falls back to prompt-JSON), low reasoning effort, cost from `usage.cost`, a per-run `Budget`; fatal errors (bad key, no credit) disable AI for the run. Model tiers live in `models.yaml`; prices and benchmark scores are fetched live from `/models`.
- **Pure math layer:** `backend/src/simulation/dynamics.py` — Cobb-Douglas, exact relaxation ODE, N-species Lotka-Volterra (RK45 → BDF → Euler), multinomial-logit share with price elasticity. No business logic.
- **Company profiles:** `backend/companies/*.oleo`.
- **Frontend:** Next.js App Router, single page `frontend/src/app/page.tsx` (model picker → setup → run). Canvas graph at `frontend/src/components/graph/ForceGraph.tsx` (the visual reference — see "Visualization" below). SSE client in `frontend/src/hooks/useSimulation.ts`, API types in `frontend/src/lib/api.ts`.

**Frontend caveat:** `frontend/AGENTS.md` warns that Next.js 16 has breaking changes from prior versions — App Router conventions in your training data may be wrong. Read `node_modules/next/dist/docs/` before writing Next.js code, and heed deprecation notices.

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
- **Pure math layer** (`backend/src/simulation/dynamics.py`): ODEs, production, competition and share models via NumPy/SciPy. No business logic, no API calls, no UI concerns.
- **Business logic layer** (`backend/src/simulation/` + `backend/src/main.py`): engine, seed/stance/library inputs, CEOs, OpenRouter client, sessions, SSE streaming.
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
- Two CEO tiers per company: `HeuristicCeo` (rule-based, weekly, free) and `AiCeo` (one LLM via OpenRouter, monthly and on the first tick of a severe shock).
- The user picks one model per run on the onboarding screen: four curated Chinese open-weight tiers (bad / mediocre / good / super good; default `z-ai/glm-5.3-flash`) or any live Claude model. Tiers are data in `models.yaml`.
- CEO persona is locked via `CeoStance` and rendered into the system prompt (inception prompting); `references_stance` is constrained to real stance fields.
- Prompts are compact (~1.5k tokens): current state, 6 months of history, rivals, active shocks, the CEO's last 3 decisions, and only the nodes that can be spawned right now.
- `Budget` enforces a per-run dollar ceiling from OpenRouter's reported cost; once spent, CEOs continue on heuristics.
- All simulation math runs without AI — AI is an overlay, not a dependency. With no `OPENROUTER_API_KEY` the heuristic CEO handles every decision.

### Target Architecture
- Per-agent persistent memory that shapes behavior across the full simulation run
- Probabilistic activation — agents fire based on domain cadence, not fixed intervals
- Department-level agents under the CEO (workforce coordinator pattern)
- Monte Carlo runner over seeds/stances with statistical comparison of outcomes

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
- **Force layout:** d3-force with charge repulsion, link attraction, center gravity, collision detection, radial layout with fixed root node (single company), and custom clustering force with roots held on screen by a gentle positioning force (multi-company).
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

- **Backend:** Python >=3.12, FastAPI, Pydantic 2, httpx (OpenRouter), typesafe-sdk (Jev), NumPy 2, SciPy, PyYAML, matplotlib + reportlab (PDF run report)
- **Frontend:** Next.js 16, React 19, d3-force (canvas-rendered), Recharts, TailwindCSS 4, TypeScript (strict). Note: Next.js 16 has breaking changes — see `frontend/AGENTS.md`.
- **AI:** OpenRouter — cheap Chinese open-weight models by default, any Claude model optional. The heuristic tier needs no key.
- **Configuration:** Universal `node_library.yaml`, `models.yaml` (CEO model tiers), JSON company profiles (`.oleo`), `CompanySeed` + `CeoStance` Pydantic models.
- **Deployment:** Local only. No cloud services. `start.py` is the single entry point.

## Project Structure

```
backend/src/main.py                      # FastAPI app, routes, sessions, SSE
backend/src/simulation/engine.py         # Company + MarketSim (tick loop)
backend/src/simulation/dynamics.py       # Pure math layer
backend/src/simulation/ceo.py            # HeuristicCeo, AiCeo, sanitize
backend/src/simulation/llm.py            # OpenRouter client, catalog, budget
backend/src/simulation/jev.py            # Jev judge for weekly rule moves
backend/src/simulation/export.py         # Saves runs (json, csv, pdf)
backend/src/simulation/report_pdf.py     # In-theme PDF run report
backend/src/simulation/report.py         # End-of-run CEO post-mortems
backend/src/simulation/library.py        # Node library + modifier buckets
backend/src/simulation/seed.py           # CompanySeed, sampler, .oleo import
backend/src/simulation/stance.py         # CeoStance + archetypes
backend/src/simulation/shocks.py         # Shock types + Poisson scheduler
backend/src/simulation/node_library.yaml # Universal node taxonomy
backend/src/simulation/models.yaml       # AI CEO model tiers
backend/companies/                       # Bundled .oleo profiles
backend/runs/                            # Saved runs (gitignored)
backend/tests/                           # pytest: engine, ceo, llm, api
frontend/src/app/page.tsx                # Model picker → setup → run
frontend/src/components/onboarding/      # ModelPicker, SimSetup
frontend/src/components/sim/             # SimView, Dashboard, TimeControls
frontend/src/components/graph/ForceGraph.tsx  # The canvas force graph — visual reference
frontend/src/hooks/useSimulation.ts      # SSE client state
frontend/src/lib/api.ts                  # API client + payload types
frontend/AGENTS.md                       # READ FIRST when editing frontend — Next.js 16 caveats
research/, references/, documentation/   # Industry research and design references (not used by code)
start.py                                 # Single entry point: key prompt, deps, dual-server launch
.env.example                             # Template; real key lives in backend/.env
```
