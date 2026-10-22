# Onyx Leopard

**Open-source, local-run business simulation framework.** Spin up N competing companies seeded from a `CompanySeed` + `CeoStance` archetype, watch them spawn org nodes, accrue cash, take shocks, and battle for market share through real Cobb-Douglas production, Lotka-Volterra competition, and multinomial-logit share allocation. AI agents at three tiers (heuristic / Haiku / Sonnet) make CEO decisions inside the tick loop.

## What It Does

1. **Pick a seed and stance archetype** — `solo_founder`, `small_team`, `venture_funded`, `enterprise` for the company shape; `bootstrap`, `founder_operator`, `venture_growth`, `consolidator`, `turnaround` for the CEO persona. Sample one via API, or load a preloaded `.oleo` profile.
2. **Run the simulation** — Hit play. Each tick (= 1 simulated day) runs production economics, derives bridge modifiers from spawned nodes, allocates demand via multinomial logit, updates Lotka-Volterra market_strength, and asks the orchestrator triad whether any nodes should spawn or retire.
3. **Watch the graph** — Companies render as force-directed canvas clusters in the browser, with nodes spawning live as the orchestrator hires roles, opens locations, and adds revenue streams.
4. **Toggle AI CEO** — With the toggle on, Haiku-tactical and Sonnet-strategic tiers fire on cadence (every 30 / 90 ticks). With it off, only the free heuristic tier runs.
5. **Shocks** — Poisson-arrival shocks (market crash, key employee departure, supplier failure, etc.) inject through the `ShockScheduler` and propagate into the math each tick.

## Preloaded Companies

Four real-world company profiles ship ready to simulate, each carrying a SHA256 checksum that the loader verifies on every read:

| File | Company |
|------|---------|
| `amazon.oleo` | Amazon |
| `apple.oleo` | Apple |
| `microsoft.oleo` | Microsoft |
| `spacex.oleo` | SpaceX |

## Quick Start

**Prerequisites:** [Python 3.12+](https://www.python.org/downloads/), [Node.js 18+](https://nodejs.org/), [pnpm](https://pnpm.io/installation)

```bash
git clone https://github.com/wdorman-tech/Onyx-Leopard.git
cd Onyx-Leopard
python start.py          # installs deps, prompts for API key, launches both servers
```

Open [http://localhost:3000](http://localhost:3000) when it's ready. The script will prompt for your [Anthropic API key](https://console.anthropic.com/) on first run — you can also set it later in `backend/.env`.

Without an API key the simulation runs in pure-math mode (the heuristic tier handles every decision, no LLM calls). With a key, Claude Haiku powers the tactical tier and Claude Sonnet powers the strategic tier + critic.

## Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | Next.js 16, React 19, d3-force (canvas-rendered), Recharts, TailwindCSS 4 |
| Backend API | FastAPI, Python 3.12+, Pydantic 2, SSE streaming |
| Math layer | NumPy 2, SciPy (RK45 → BDF → Euler), Mesa 3 |
| AI Agents | Anthropic SDK direct (Haiku tactical, Sonnet strategic + critic) |
| Math Models | Cobb-Douglas production, Lotka-Volterra competition, logistic growth ODEs, multinomial-logit share |
| Data | `.oleo` profile format (JSON + SHA256 checksum verification) |

## Project Structure

```
├── frontend/src/
│   ├── app/                 # Next.js app router (page.tsx + /api/companies)
│   ├── components/
│   │   ├── graph/           # ForceGraph (canvas-rendered d3-force) + NodeTooltip
│   │   ├── modes/           # ModeSelector, UnifiedSetup, UnifiedSimView
│   │   ├── simulation/      # UnifiedDashboard, CompanyDashboard, MarketDashboard, TimeControls
│   │   └── ui/              # Shared UI components
│   ├── hooks/               # useUnifiedSimulation (v2 SSE client)
│   ├── lib/                 # apiV2.ts (v2 fetch client)
│   └── types/               # TypeScript types for graph & v2 payloads
├── frontend/public/
│   └── companies/           # Preloaded .oleo profiles (Amazon, Apple, Microsoft, SpaceX)
├── backend/src/
│   ├── main.py              # FastAPI app, CORS, /health, mounts v2 router
│   ├── routes/              # FastAPI endpoints (simulation_v2.py — SSE streaming)
│   ├── simulation/          # Business logic
│   │   ├── unified_v2.py        # MultiCompanySimV2 + CompanyAgentV2
│   │   ├── orchestrator.py      # HeuristicOrchestrator + TacticalOrchestrator + StrategicOrchestrator + OrchestratorBundle
│   │   ├── critic.py            # CriticAgent — stance-alignment scoring
│   │   ├── seed.py              # CompanySeed + archetype samplers
│   │   ├── stance.py            # CeoStance + archetype samplers
│   │   ├── library_loader.py    # NodeLibrary loader
│   │   ├── bridge.py            # Modifier derivation from spawned nodes
│   │   ├── shocks.py            # ShockScheduler + apply_active_shocks
│   │   ├── replay.py            # CostTracker + Transcript (record / replay / off modes)
│   │   ├── monte_carlo_v2.py    # Parallel run sweep
│   │   ├── market/              # Standalone MarketEngine + presets
│   │   └── node_library.yaml    # Universal node taxonomy
│   └── biosim/              # Pure math layer (no business logic)
│       ├── math/                # production.py (Cobb-Douglas), competition.py (LV), solver.py (RK45→BDF→Euler), growth.py
│       ├── engine/              # tick.py — vectorized Mesa-style step
│       ├── agents/              # Mesa wrapper
│       └── types/               # StateArrays (struct-of-arrays) + enums
└── backend/tests/           # pytest suites (test_unified_v2, test_growth_invariants, test_orchestrator_*, test_market_*, test_biosim_integration, ...)
```

## How the Simulation Works

Each tick (= 1 simulated day) inside `MultiCompanySimV2.step` runs:

1. **TAM growth** — TAM grows by `market_growth_rate` per tick.
2. **Per-company attraction** — Compute quality / marketing from each company's spawned-node bridge aggregate, apply per-company shock modifiers (`market_share_mult`, `competitor_pressure_mult`), and multiply by Lotka-Volterra `market_strength`.
3. **Share allocation** — Multinomial logit `s_i ∝ q_i^β * m_i^α`.
4. **Per-company tick** — Each `CompanyAgentV2` runs its own step: shocks → bridge aggregation → Cobb-Douglas production cap → satisfaction ODE → revenue / cost / cash → orchestrator triad (heuristic → tactical → strategic + critic) → apply decisions.
5. **Lotka-Volterra update** — Cross-company `market_strength` evolves with the N-species LV system (`build_competition_matrix` + `step_competition`).
6. **Death check** — Companies insolvent for `INSOLVENT_TICKS_TO_DEATH` consecutive ticks die; the last death emits an `all_dead` payload.

The orchestrator triad runs at different cadences: heuristic every 7 ticks (free), tactical every 30 ticks (Haiku), strategic every 90 ticks (Sonnet) or on any severe-shock interrupt. The critic agent scores every strategic decision against the locked `CeoStance`. The `CostTracker` enforces a per-run dollar ceiling; the `Transcript` records every LLM call so runs can be deterministically replayed.

## The .oleo Format

`.oleo` is a JSON-based company profile format with SHA256 checksum verification — the loader in `frontend/src/app/api/companies/route.ts` re-computes the SHA256 over the canonicalized (sorted-key, separator-free) JSON and rejects any file whose recorded `checksum` does not match. Each file contains company identity, organization structure, financials, market profile, operations, strategy, and simulation parameters; the loader projects those into a `CompanySeed` for the v2 engine.

## Configuration

| Variable | Location | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | `backend/.env` | Required for tactical (Haiku) + strategic (Sonnet) + critic (Sonnet) tiers. Without it, only the heuristic tier runs and the sim is pure-math. |
| `ONYX_TRANSCRIPT_BASE` | env var (optional) | Override the sandbox directory under which `transcript_dir` paths resolve. Defaults to `backend/transcripts/`. |

Per-request controls on `POST /api/v2/simulate/start`:

- `ai_ceo_enabled` — Toggle the Haiku/Sonnet tiers on/off (default: on). When off, only the heuristic tier fires.
- `cost_ceiling_usd` — Dollar ceiling per simulation run (default: $10.00, must be > 0).
- `rng_seed` — Deterministic run seed. Identical `rng_seed` ⇒ identical `sim_id`, per-company RNG draws, and LV competition matrix.
- `transcript_dir` — Sandboxed relative path under `ONYX_TRANSCRIPT_BASE` for recording the LLM transcript. Absolute paths and `..` traversal are rejected.
- `seeds` / `stances` — Optional per-company arrays (length must equal `num_companies`) so each company can have a distinct seed and CEO persona. Otherwise the single `seed`/`stance` is cloned across all companies.

## Development

```bash
# Backend
cd backend && uv sync --group dev
uv run pytest -m "not slow"     # full suite, skip multi-year tests
uv run ruff check .              # lint gate (currently green)

# Frontend
cd frontend && pnpm install && pnpm dev
pnpm build                       # production build doubles as type-check gate
pnpm lint                        # eslint
```

## License

[MIT](LICENSE)
