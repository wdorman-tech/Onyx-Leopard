# Onyx Leopard

**Open-source, local-run business simulation.** Clone one company N times, hand each copy a different AI CEO persona, and watch them compete for a shared market, day by day. Company growth, competition and market share come from established models (Cobb-Douglas production, Lotka-Volterra competition, multinomial-logit share). The CEOs are LLMs on OpenRouter, defaulting to cheap Chinese open-weight models, with any Claude model one click away.

## Quick Start

**Prerequisites:** [Python 3.12+](https://www.python.org/downloads/), [Node.js 18+](https://nodejs.org/), [pnpm](https://pnpm.io/installation). [uv](https://docs.astral.sh/uv/) is used when installed.

```bash
git clone https://github.com/wdorman-tech/Onyx-Leopard.git
cd Onyx-Leopard
python start.py          # installs deps, asks for your OpenRouter key, launches both servers
```

Open [http://localhost:3000](http://localhost:3000). On first run `start.py` asks for an [OpenRouter key](https://openrouter.ai/keys), saves it to `backend/.env`, and checks it with OpenRouter. Press Enter to skip: the simulation still runs, with free rule-based CEOs.

## Using It

1. **Pick the CEO model.** Four curated Chinese tiers (prices and benchmark scores fetched live from OpenRouter), or any Claude model:

   | Tier | Model | Why |
   |------|-------|-----|
   | Bad | `qwen/qwen3.5-9b` | Tiny and nearly free. Fun to watch go bankrupt. |
   | Mediocre | `deepseek/deepseek-v4-flash` | Fast, cheap, misses second-order effects. |
   | **Good (default)** | `z-ai/glm-5.3-flash` | Near-frontier reasoning; a 5-year, 4-CEO run costs about $0.03. |
   | Super good | `xiaomi/mimo-v2.6-pro` | Strongest open-weight model available. |

   Edit `backend/src/simulation/models.yaml` to change the tiers; any OpenRouter model id works.
2. **Set up the run.** Use a bundled company (Amazon, Apple, Microsoft, SpaceX) or sample one, give each competitor a CEO archetype (founder-operator, venture growth, bootstrapper, consolidator, turnaround), and pick a duration, a hard dollar budget, and whether market shocks fire.
3. **Watch.** Each company is a cluster in the force graph; nodes appear as CEOs hire, open locations and add revenue streams. The dashboard shows share, cash, every CEO decision with its reasoning, and AI spend. Each CEO writes a post-mortem at the end.

## How a Day Works

`MarketSim.step` in `backend/src/simulation/engine.py`:

1. TAM compounds at the seed's growth rate; market-wide shocks arrive (Poisson) or expire.
2. Share of customer spend per company: multinomial logit over satisfaction, marketing reach, price (elasticity 1.5) and Lotka-Volterra market strength.
3. Each company serves `min(demand, physical capacity, Cobb-Douglas production)`, books revenue and costs, and updates cash. A month of negative cash is bankruptcy.
4. Market strength evolves under N-species Lotka-Volterra competition.
5. CEOs decide. The heuristic CEO acts weekly: it cuts costs in a cash crisis, expands when capacity binds, and hires when production binds. The AI CEO acts monthly, and early on severe shocks: it spawns or retires nodes, opens locations, and moves price, marketing and capital. Every decision is sanitized against the node library and engine bounds.

AI calls use strict JSON-schema output (node keys are enums, so weak models can't invent them), low reasoning effort, and a compact prompt of about 1.5k tokens. Spend is tracked from OpenRouter's reported cost; when the budget runs out, CEOs fall back to heuristics.

## Project Structure

```
backend/
  companies/*.oleo            # bundled company profiles
  src/main.py                 # FastAPI app: config, models, companies, start/stream/control
  src/simulation/
    engine.py                 # Company + MarketSim (the tick loop)
    dynamics.py               # pure math: Cobb-Douglas, LV, logit share, relaxation ODE
    ceo.py                    # HeuristicCeo, AiCeo, decision sanitizer
    llm.py                    # OpenRouter client, model catalog, run budget
    report.py                 # end-of-run CEO post-mortems
    library.py                # node library loader + modifier buckets
    seed.py / stance.py       # company seeds, CEO personas
    shocks.py                 # shock types + Poisson scheduler
    node_library.yaml         # universal node taxonomy (industry-agnostic)
    models.yaml               # AI CEO model tiers
  tests/                      # pytest (engine, CEOs, OpenRouter client, API)
frontend/src/
  app/page.tsx                # model picker → setup → run
  components/onboarding/      # ModelPicker, SimSetup
  components/sim/             # SimView, Dashboard, TimeControls
  components/graph/           # ForceGraph (canvas d3-force) + NodeTooltip
  hooks/useSimulation.ts      # SSE client state
  lib/api.ts                  # API client + payload types
start.py                      # one-command setup and launch
```

## Configuration

| Variable | Where | Purpose |
|----------|-------|---------|
| `OPENROUTER_API_KEY` | `backend/.env` | Enables AI CEOs. `start.py` writes it. |
| `OPENROUTER_BASE_URL` | env or `backend/.env` | Optional: point at a proxy or local stand-in. |

## Development

```bash
cd backend
uv sync --group dev
uv run pytest            # ~2s
uv run ruff check src tests

cd frontend
pnpm install && pnpm dev
pnpm lint && pnpm build  # build doubles as the type-check
```

## License

[MIT](LICENSE)
