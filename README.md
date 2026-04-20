# Onyx Leopard

**AI-powered business strategy simulator.** Build a company from scratch, import one from SEC EDGAR, or load a preloaded profile (.oleo). Watch it come to life as an interactive org graph, then run a multi-agent biological simulation where every department makes autonomous decisions — hiring, investing, cutting costs — driven by real production economics, ecological competition, and AI agents that handle the strategic edge cases.

## What It Does

1. **Build your company** — Use the guided questionnaire, paste a description, upload documents, or pull real financials from SEC EDGAR. Claude turns it into a structured company profile with departments, revenue streams, and simulation parameters.
2. **Visualize the org** — Departments, teams, roles, and their relationships render as an interactive flow graph you can explore and refine through chat.
3. **Run the simulation** — Hit play. An 8-phase tick engine drives each simulated week: Cobb-Douglas production economics, logistic growth ODEs, Lotka-Volterra competition, and 12 AI agents per company making autonomous strategic decisions via a 4-tier hybrid engine.
4. **Inject chaos** — Drop events mid-simulation ("supplier goes bankrupt", "competitor launches rival product") and watch how the AI agents across all companies respond and adapt.
5. **Compare scenarios** — Fork simulations, change parameters, and compare outcomes side by side.

## Preloaded Companies

Four real-world company profiles ship ready to simulate:

| File | Company |
|------|---------|
| `amazon.oleo` | Amazon |
| `apple.oleo` | Apple |
| `microsoft.oleo` | Microsoft |
| `spacex.oleo` | SpaceX |

Import any `.oleo` file to load a complete company profile with org structure, financials, market data, operations, strategy, and simulation parameters. Export your own companies as `.oleo` to share or reload later.

## Quick Start

**Prerequisites:** [Python 3.12+](https://www.python.org/downloads/), [Node.js 18+](https://nodejs.org/), [pnpm](https://pnpm.io/installation)

```bash
git clone https://github.com/wdorman-tech/Onyx-Leopard.git
cd Onyx-Leopard
python start.py          # installs deps, prompts for API key, launches servers
```

Open [http://localhost:3000](http://localhost:3000) when it's ready. The script will prompt for your [Anthropic API key](https://console.anthropic.com/) on first run — you can also set it later in `backend/.env`.

Without an API key the simulation runs in pure-math mode (Tier 0/1 only — no LLM calls). With a key, Claude Sonnet powers the company builder and Claude Haiku/Sonnet drive department-level agent decisions.

## Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | Next.js 16, React 19, React Flow (xyflow), Recharts, TailwindCSS 4 |
| Backend API | FastAPI, Python 3.12+, Pydantic 2 |
| Simulation Engine | Mesa 3, NumPy (vectorized struct-of-arrays), SciPy (ODE solvers) |
| AI Agents | CAMEL-AI framework, Claude Sonnet + Haiku via Anthropic API |
| Math Models | Cobb-Douglas production, Lotka-Volterra competition, logistic growth ODEs |
| Data | SEC EDGAR via edgartools, .oleo profile format (JSON + SHA256 checksum) |

## Project Structure

```
├── frontend/src/
│   ├── app/                 # Next.js app router
│   ├── components/
│   │   ├── chat/            # Chat-based company building
│   │   ├── flow/            # React Flow org graph canvas
│   │   ├── simulation/      # Simulation controls & real-time dashboard
│   │   ├── onboarding/      # Questionnaire, EDGAR lookup
│   │   ├── toolbar/         # Import/Export .oleo buttons
│   │   └── ui/              # Shared UI components
│   ├── hooks/               # useSimulation, useOnboarding, useScenarios
│   ├── lib/                 # API client, layout utilities
│   └── types/               # TypeScript types for graph & profile
├── frontend/public/
│   └── companies/           # Preloaded .oleo profiles (Amazon, Apple, Microsoft, SpaceX)
├── backend/src/
│   ├── agents/              # Claude parsers, questionnaire, graph generator, EDGAR client
│   ├── simulation/          # Session manager, state, agent prompts
│   ├── routes/              # FastAPI endpoints (company, import/export, documents)
│   └── schemas.py           # Pydantic models (CompanyProfile, OleoExport, etc.)
├── src/biosim/
│   ├── agents/              # Mesa agents + CAMEL-AI department agents
│   │   ├── mesa_model.py    # BioSimModel (Mesa wrapper)
│   │   ├── department_agent.py  # CAMEL-AI ChatAgent wrappers (12 departments)
│   │   ├── prompts.py       # System prompt templates
│   │   ├── tools.py         # Simulation action tools (hire, fire, budget, etc.)
│   │   └── llm_backend.py   # Agent pool manager
│   ├── engine/              # 8-phase tick loop
│   │   ├── tick.py          # TickEngine (sense → ODE → agents → compete → grow → emit)
│   │   ├── state_manager.py # StateArrays lifecycle + history
│   │   ├── decision_router.py  # 4-tier hybrid decision routing
│   │   └── heuristics.py    # Tier 1 rule-based decisions (12 departments)
│   ├── math/                # Vectorized computation
│   │   ├── production.py    # Cobb-Douglas: Y = A * K^α * L^β
│   │   ├── competition.py   # N-species Lotka-Volterra
│   │   └── solver.py        # Batched ODE solver (RK45 → BDF → Euler)
│   └── types/               # State arrays, config, enums, decision types
└── references/              # PRD, research docs
```

## How the Simulation Works

Each tick (= 1 simulated week) runs 8 phases:

1. **Sense** — Update market share signals
2. **Solve ODEs** — Cobb-Douglas production → revenue; logistic growth ODE → firm size and cash
3. **Agent Decisions** — 4-tier routing classifies each department's situation by novelty. Tier 0 (ODE, ~60%) and Tier 1 (heuristics, ~20%) run at zero cost. Tier 2 (Haiku, ~10%) and Tier 3 (Sonnet, ~10%) make LLM calls for genuinely novel situations.
4. **Interactions** — Lotka-Volterra N-species competition between firms
5. **Growth/Division** — Department expansion when growth threshold exceeded; agent decisions (hiring, budget changes, capacity investment) are applied
6. **Environment Update** — Market condition updates
7. **Selection** — Remove companies insolvent for 3+ consecutive ticks
8. **Emit** — Update health scores, reinvest capital, push state to frontend via SSE

**Cost:** ~$0.30–0.80 for a 50-tick demo with 3 companies. Budget caps prevent runaway spend.

## The .oleo Format

`.oleo` is a JSON-based company profile format with SHA256 checksum verification. Each file contains:

- **Company identity** — Name, industry, NAICS code, business model, stage
- **Organization structure** — Departments, teams, roles, headcount
- **Financials** — Revenue, COGS, margins, assets, debt, ratios
- **Market profile** — TAM/SAM/SOM, competitors, pricing
- **Operations** — Production model, suppliers, inventory
- **Strategy** — Objectives, initiatives, risks, moats
- **Simulation parameters** — Cobb-Douglas coefficients, ODE initial conditions, seasonality

Import via the toolbar or create from scratch through the chat questionnaire.

## Configuration

| Variable | Location | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | `backend/.env` | Required for company building (Sonnet) and simulation agents (Haiku/Sonnet) |

Agent behavior is configured in `SimConfig.agent_config`:
- `enabled` — Toggle AI agents on/off (default: off)
- `novelty_threshold` — L2 norm delta to escalate decision tier (default: 0.15)
- `cost_budget_per_run` — Dollar cap for LLM calls (default: $5.00)

## Development

```bash
# Backend
cd backend && uv sync --group dev
uv run pytest -v
uv run ruff check src/

# Simulation engine
cd .. && uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/

# Frontend
cd frontend && pnpm install && pnpm dev
```

## License

[MIT](LICENSE)
