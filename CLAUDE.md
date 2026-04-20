# Onyx Leopard — Guiding Principles

Every build decision, feature addition, and refactor must align with these principles. This is the constitution of the project. Read it before writing code.

## What This Is

Onyx Leopard is an open-source, local-run business simulation framework. It models any business as a node-based growth system where AI agents operate departments, make decisions, and react to market conditions. The simulation produces emergent outcomes — not scripted results.

**The end goal:** Run 10+ parallel simulations of slightly different business configurations simultaneously, Monte Carlo style, to determine optimal structure, strategy, and resilience. Stress-test a business against market shocks, competitive pressure, and internal failures before it exists in the real world.

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
- **Pure math layer** (`src/biosim/`): ODEs, growth models, competition models via Mesa/NumPy/SciPy. No business logic, no API calls, no UI concerns.
- **Business logic layer** (`backend/src/`): FastAPI routes, session management, industry configs, AI agent orchestration, SSE streaming. Contains its own simulation engines (`UnifiedEngine`, `MarketEngine`, `GrowthEngine`) that currently implement market math directly. Long-term, shared math should consolidate into the pure math layer.
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
- CEO agents use Claude Sonnet, called on a fixed interval (every 182 ticks / ~6 months sim time)
- Agents observe company state and market conditions, return structured JSON decisions
- Decisions apply as parameter mutations (price, cost, expansion pace, marketing boost)
- CEO decision history (last 2 decisions) is included in prompts for continuity
- All simulation math runs without AI — AI is an overlay, not a dependency

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

- **Backend:** Python >=3.12, FastAPI, NumPy, SciPy, Pydantic 2, camel-ai (multi-agent framework)
- **Pure math layer:** Mesa 3, NumPy, SciPy, NetworkX (in `src/biosim/`, separate from backend)
- **Frontend:** Next.js 16, React 19, d3-force (canvas-rendered), Recharts, TailwindCSS 4, TypeScript (strict)
- **AI:** Anthropic Claude API — currently Sonnet only; target is Sonnet (executive) + Haiku (operational)
- **Configuration:** YAML industry specs, JSON company profiles (.oleo format)
- **Deployment:** Local only. No cloud services. `start.py` is the single entry point.

## Project Structure

```
backend/src/simulation/          # Business logic + engine orchestration
backend/src/simulation/market/   # Market competition engine
backend/src/simulation/industries/  # YAML industry configs
backend/src/routes/              # FastAPI API routes
src/biosim/                      # Pure math engine (NumPy/SciPy)
frontend/src/                    # Next.js frontend
documentation/                   # Architecture docs
research/industries/             # Industry research (sourced)
tests/                           # Test suites
```
