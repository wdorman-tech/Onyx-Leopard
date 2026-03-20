# Onyx Leopard

**AI-powered business strategy simulator.** Describe your company, watch it come to life as an interactive org graph, then run a multi-agent simulation where every department makes autonomous decisions — hiring, investing, cutting costs — based on market conditions and events you throw at them.

## What It Does

1. **Build your company** — Use the guided questionnaire, paste a description, upload documents, or pull data from SEC filings via EDGAR. Claude turns it into a structured company profile.
2. **Visualize the org** — Departments, teams, roles, revenue streams, and their relationships render as an interactive graph you can refine through chat.
3. **Run the simulation** — Hit play. Each node becomes a Claude-powered agent that analyzes its metrics, reacts to market outlook, and makes strategic decisions every tick. Watch headcount, revenue, and costs evolve in real time.
4. **Inject chaos** — Drop events mid-simulation ("supplier goes bankrupt", "competitor launches rival product") and see how your org adapts.
5. **Compare scenarios** — Fork simulations, change the outlook, and compare outcomes side by side.

## Quick Start

**Prerequisites:** Node.js 18+, Python 3.12+, an [Anthropic API key](https://console.anthropic.com/)

```bash
# Backend
cd backend
uv venv && uv sync          # or: python -m venv .venv && pip install -e .
cp .env.example .env         # add your ANTHROPIC_API_KEY
uvicorn src.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) and follow the onboarding flow.

## Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | Next.js 16, React 19, React Flow, Recharts, TailwindCSS 4 |
| Backend | FastAPI, Python 3.12+, Pydantic 2, CAMEL-AI |
| AI | Claude Sonnet (questionnaire/parsing), Claude Haiku (simulation agents) |
| Streaming | Server-Sent Events for real-time tick updates |

## Project Structure

```
├── frontend/src/
│   ├── app/              # Next.js app router (single-page app)
│   ├── components/       # Chat, flow canvas, simulation controls, onboarding
│   ├── hooks/            # useSimulation, useOnboarding, useScenarios
│   ├── lib/              # API client, layout utilities
│   └── types/            # TypeScript types for graph & profile
├── backend/src/
│   ├── agents/           # Claude-powered parsers, questionnaire, graph generator
│   ├── simulation/       # Engine, session manager, state, agent prompts
│   ├── routes/           # FastAPI endpoints
│   └── schemas.py        # All Pydantic models
└── references/           # Research docs
```

## How the Simulation Works

Each org node gets its own Claude Haiku agent with a tailored system prompt. Every tick, all agents run in parallel — they see global metrics, their own node's state, the current market outlook, and any injected events. Each agent returns a structured decision (hire, invest, cut costs, etc.) that gets applied to the simulation state. Results stream to the frontend via SSE so you watch it unfold live.

**Cost:** ~$0.08 for a full 50-tick run with 20 nodes. Haiku keeps it cheap.

## Configuration

| Variable | Location | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | `backend/.env` | Required for all AI features |

## License

Proprietary — Black Lily LLC
