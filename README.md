# Onyx Leopard

**Open-source, local-run business simulation.** Clone one company N times, hand each copy a different AI CEO persona, and watch them compete for a shared market, day by day. Company growth, competition and market share come from established models (Cobb-Douglas production, Lotka-Volterra competition, multinomial-logit share). The CEOs are LLMs on OpenRouter, defaulting to cheap Chinese open-weight models, with any Claude model one click away.

## Quick Start

**Prerequisites:** [Python 3.12+](https://www.python.org/downloads/), [Node.js 18+](https://nodejs.org/), [pnpm](https://pnpm.io/installation). [uv](https://docs.astral.sh/uv/) is used when installed.

```bash
git clone https://github.com/wdorman-tech/Onyx-Leopard.git
cd Onyx-Leopard
python start.py          # installs deps, asks for your OpenRouter key, launches both servers
```

Open [http://localhost:3000](http://localhost:3000). On first run `start.py` asks for an [OpenRouter key](https://openrouter.ai/keys), saves it to `backend/.env`, and checks it with OpenRouter. Press Enter to skip: the first screen of the app then asks for the key (it's verified with OpenRouter and saved to `backend/.env`; no restart). Without a key the simulation still runs, with free rule-based CEOs.

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
4. **Study the run.** Every finished run is saved to `backend/runs/<run>/`: an in-theme **PDF report** (leaderboard, finance charts for every company, each CEO's post-mortem and decision log, market events), `timeseries.csv` (every metric, every company, across the whole run) and `run.json` (everything, including every decision). The dashboard links straight to them.

## How a Day Works

`MarketSim.step` in `backend/src/simulation/engine.py`:

1. TAM compounds at the seed's growth rate; market-wide shocks arrive (Poisson, and a type that is already running can't stack on itself) or expire. Severity compounds the minor impact in log space, and every impact scales with the company.
2. A multinomial logit over satisfaction, marketing reach and price gives the share each company's offer earns. Its price elasticity is calibrated so the seed's starting price is the Bertrand-Nash price while demand binds (Lerner condition on the contribution margin, revenue-stream uplift and marketing included); a company that starts capacity-bound is right to raise price until settled demand meets capacity. Customer bases move toward those shares under N-species Lotka-Volterra competition, with carrying capacities set so the rest point is the logit share: customers switch over weeks, and nobody is competitively excluded at rest. Headcount buys capacity, not share; sales and marketing reach enters at the empirical selling-effort elasticity (~0.3), so a sales army wins share without wiping out rivals.
3. Category spend follows the market's average price along a linear demand curve calibrated to the seed's price elasticity at the starting price (below 1, a matched price war shrinks the market; demand grows more elastic as prices rise and stops at a choke price no customer pays, so an offer priced there wins nobody and even a lone company has a finite best price). Each company serves `min(demand, physical capacity, Cobb-Douglas production)`, books revenue and costs, and updates cash. A month of negative cash is bankruptcy.
4. CEOs decide. The heuristic CEO acts weekly: it cuts the node that saves the most net of what it earns in a cash crisis, expands when capacity binds, and hires when production binds. With a TypeSafe key, [Jev](https://typesafe.ai) judges each of those weekly moves before it happens: Python detects the setup, Jev vetoes it only when its answer falls well below its rolling baseline for that kind of move, and a cut skips Jev when holding off would run out of cash inside the 90-day horizon. The AI CEO acts monthly, and early on severe shocks: it spawns or retires nodes, opens locations, and moves price, marketing and capital. It sees its gross and contribution margins, its own-price elasticity, settled demand against capacity, a best-response price (the list price that maximizes its settled profit if rivals hold, found by a bounded search) and every active shock. Marketing reach uses the (1 + intensity) transform, so cutting marketing to zero costs about a third of reach, not all of it. Every decision is sanitized against the node library and engine bounds.

AI calls use strict JSON-schema output (node keys are enums, so weak models can't invent them), low reasoning effort, and a compact prompt of about 2-3k tokens. Spend is tracked from OpenRouter's reported cost and paced over the run: when a budget can't cover a monthly cadence, AI rounds spread out instead of stopping partway; heuristics cover the gaps.

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
    jev.py                    # Jev judge for the weekly rule CEO (TypeSafe)
    report.py                 # end-of-run CEO post-mortems
    export.py                 # saves runs: run.json, timeseries.csv, report.pdf
    report_pdf.py             # the in-theme PDF run report
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
| `TYPESAFE_API_KEY` | `backend/.env` | Optional: enables the Jev weekly judge. Without it, rules act alone. |
| `BACKEND_URL` / `NEXT_PUBLIC_API_URL` | frontend env | Optional: run a second copy on other ports (default `http://localhost:8000`). |

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
