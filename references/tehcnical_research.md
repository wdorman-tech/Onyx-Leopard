# Architecting an LLM-powered economic simulation engine

**A weekly-tick business simulation combining CAMEL-AI OASIS patterns, Mesa-style agent hierarchies, mathematical economic models, and PyQt real-time visualization can achieve production viability by adopting a hybrid rule-based/LLM architecture that reduces token costs by 97–99%.** This report synthesizes research across six domains to provide a complete architectural blueprint for a Python desktop application that simulates firm-level business growth with configurable competitor detail levels. The core insight: OASIS provides the LLM-agent scaffolding and communication patterns, Mesa provides proven economic ABM design patterns, and a tiered decision architecture makes the system economically feasible at scale.

---

## 1. OASIS provides LLM-agent scaffolding, not economic modeling

CAMEL-AI's **OASIS** (Open Agent Social Interaction Simulations) is an open-source framework for simulating up to one million LLM-powered agents on social media platforms. Released November 2024 (arXiv 2411.11581), it lives at `github.com/camel-ai/oasis` with **2.3k stars** and Apache 2.0 licensing. Its architecture comprises five components: an **Environment Server** (SQLite-backed state), a **Recommendation System** (content routing), an **Agent Module** (extending CAMEL's `ChatAgent`), a **Time Engine** (probabilistic activation with 24-hour activity vectors), and a **Scalable Inferencer** (async multi-GPU inference queues).

The simulation loop follows a **PettingZoo-style interface**: `env = oasis.make(agent_graph, platform, database_path)` → `await env.reset()` → `await env.step(actions)` per tick → `await env.close()`. Each step advances the simulation clock, refreshes agent observations, processes LLM decisions via tool-calling, and updates the SQLite state. Agents decide actions through **function tools** exposed to the LLM — the model selects which tool to call and with what arguments. This means only models with tool-calling support work (GPT-4o-mini, Claude Sonnet, Llama 3+ with tool support, Qwen via VLLM).

**Key architectural patterns worth adopting:**

The `SocialAgent` extends `ChatAgent` with a memory module (encountered data, previous actions, reasoning) and an action module (available operations). The `AgentGraph` container manages all agents and relationships. The `Channel` class provides async message queues between agents and the environment server with UUID-based request tracking. Actions are typed via an `ActionType` enum with **28+ predefined actions**, and new actions can be added. The `ManualAction` vs `LLMAction` distinction is critical: manual actions execute predefined logic (zero token cost), while LLM actions invoke the model for autonomous decision-making. Custom `FunctionTool` instances can be attached to agents for domain-specific capabilities.

**Limitations for economic simulation are significant.** OASIS is fundamentally designed for social media — its database schema tracks posts, comments, likes, and follows rather than financial accounts, inventories, or market prices. The recommendation system routes content, not market signals. Inter-agent communication is mediated entirely through the platform (Agent A posts → RecSys routes to Agent B's feed), with no direct negotiation or transaction protocol. Adapting OASIS to economics requires replacing the Platform's database schema, creating custom ActionTypes (`PLACE_ORDER`, `SET_PRICE`, `HIRE_WORKER`, `INVEST`), implementing market-clearing as a custom RecSys replacement, and adding economic state tables. The PettingZoo-style `env.step()` loop and the `LLMAction`/`ManualAction` duality are the most transferable patterns — the social media specifics should be treated as reference implementations rather than direct dependencies.

---

## 2. Mesa and ACE literature define the economic agent architecture

The **Mesa framework** (v3.x, actively developed toward v4.0) provides the canonical Python ABM architecture. Its core consists of `Model`, `Agent`, and `AgentSet` classes, with deprecated schedulers replaced by flexible activation patterns. The critical design choice for economic models is **staged activation by type**: each simulation step processes agent types in sequence (firms adjust prices → firms hire/fire → households search for jobs → households consume → firms produce → payments settle). Mesa's `AgentSet` supports this via `model.agents_by_type[Firm].shuffle_do("adjust_price")`, which shuffles within each type to avoid systematic ordering bias while maintaining logical phase ordering.

For non-spatial economic models, Mesa requires no grid or space — agents interact through direct references, `AgentSet` queries, and optional `Network` connections for supply chains. The `DataCollector` class captures model-level metrics (GDP, unemployment, Gini coefficient) and agent-level variables (wealth, inventory, market share) at each step, outputting pandas DataFrames for analysis. Batch running via `batch_run()` enables Monte Carlo parameter sweeps with multiprocessing.

**Three reference architectures from computational economics literature define the design space:**

The **Lengnick baseline model** (2013, JEBO) demonstrates that ~1,000 households and ~100 firms with simple adaptive rules reproduce Phillips curves, Beveridge curves, and endogenous business cycles without exogenous shocks. Its architecture uses months divided into StartOfMonth (price/wage adjustments, job search) → 21 working days (production, consumption) → EndOfMonth (payments, profit distribution). Firms adjust prices via simple heuristics: inventory too low → raise price by random 0–5%; inventory too high → lower price. This minimal design is the strongest starting point for a business growth simulator.

The **Dosi et al. K+S model** adds Schumpeterian innovation through a two-sector industry (capital-goods firms that innovate/imitate technologies, consumption-goods firms that produce using capital and labor). Innovation is stochastic: R&D spending generates a probability of discovering new technology parameters, creating realistic creative destruction dynamics. Market clearing uses **competitiveness-based probabilistic demand allocation** — each firm's market share evolves based on relative prices and unfilled demand, not centralized auction. This pattern is directly applicable for modeling competitive dynamics with heterogeneous firms.

The **Axtell firm formation model** demonstrates emergence at massive scale: 120 million worker-agents self-organizing into ~6 million firms via team production with increasing returns, reproducing Zipf-distributed firm sizes as an emergent property. The key insight: firms are coalitions, not fixed entities. For a configurable competitor detail system, this suggests modeling the user's firm with full financial detail while representing competitors as statistical aggregates whose behavior emerges from simpler rules.

**Recommended agent hierarchy for the target application:**

- **FirmAgent** (full detail for user's company): revenue, costs (COGS, SG&A, R&D), capital stock, labor force, inventory, cash, debt, production function, pricing strategy, investment policy
- **CompetitorAgent** (configurable detail): at minimum — price, output, market share, cost structure; at maximum — full FirmAgent detail with LLM strategic reasoning
- **MarketAgent** (singleton): aggregates supply/demand, computes market-clearing price, tracks industry-level metrics
- **MacroAgent** (singleton): generates exogenous economic conditions via regime-switching, modulates demand growth, interest rates, input costs

Market clearing should use **decentralized search-and-match** (Lengnick pattern) for high-detail mode or **posted-price with probabilistic demand allocation** (K+S pattern) for computational efficiency. Stock-flow consistency is essential: every payment must have a matching receipt, and money must be conserved across all agents.

---

## 3. Token optimization makes LLM-agent simulation economically viable

Running a naive LLM simulation — 10 agents × 200 ticks × 3 decisions per tick at ~3,000 tokens per call using GPT-4o — costs roughly **$90–180 per simulation run**. A layered optimization strategy reduces this to **$1–2.50**, achieving 97–99% savings through six techniques applied in priority order.

**The hybrid rule-based/LLM architecture is the single highest-impact optimization.** For a business simulation, **60–80% of decisions are routine**: price adjustments within ±5% bands, inventory reordering below thresholds, standard capacity utilization, following an established strategic plan. These should execute via deterministic rule engines at zero token cost. The LLM handles only novel situations (new competitor entry, market regime change), strategic pivots (entering/exiting markets), and complex multi-factor reasoning. A lightweight decision classifier routes each decision to the appropriate tier:

- **Tier 1 — Rule-based (zero cost):** EOQ reordering, formulaic price adjustments, wage indexing, depreciation schedules
- **Tier 2 — Small model (GPT-4o-mini/Haiku at $0.15–0.50/M tokens):** moderate pricing decisions, competitive responses, quarterly plan adjustments
- **Tier 3 — Large model (GPT-4o/Sonnet at $2.50–5/M tokens):** strategic pivots, novel competitive situations, long-term investment decisions

**State compression reduces per-call tokens by 60–80%.** Replace narrative agent state descriptions with structured JSON key-value representations: `{"cash": 245000, "inventory": 1200, "price": 42.50, "market_share": 0.18, "competitor_prices": [39, 45, 41]}`. History should use hierarchical compression: raw tick events → filtered key events → periodic summaries → rolling strategic summary. The "Affordable Generative Agents" paper (Yu et al., Tencent, TMLR 2024) demonstrated reducing costs to **31.1% of baseline** through a Lifestyle Policy that caches agent-environment interaction patterns and reuses stored plans for similar situations without any LLM call.

**Semantic decision caching provides compounding savings.** Embed agent state + decision type into vectors using lightweight models (e5-small-v2, ~$0.0001 per embedding), search a FAISS index for similar past decisions, and return cached results when similarity exceeds a threshold (0.80–0.95). Cache keys should be quantized: round cash to nearest $10K, inventory to nearest 100 units, creating finite buckets. Invalidation triggers on major events (new competitor, demand shock) or after N ticks.

Additional optimizations include **OpenAI/Anthropic Batch API** (50% cost discount for asynchronous processing — ideal since simulation ticks aren't real-time), **structured JSON output schemas** (60–80% output token reduction, eliminates parsing retries), **prompt caching** (Anthropic offers 90% reduction on cached prefix tokens — place the stable system prompt and tool definitions at the beginning), and `temperature=0` for deterministic operational decisions enabling more effective response caching.

---

## 4. PyQt architecture demands pyqtgraph and the worker-object pattern

Real-time simulation visualization requires separating the simulation engine from the GUI thread. The canonical PyQt pattern uses **moveToThread with a worker QObject**: the `SimulationWorker(QObject)` runs in a `QThread`, communicates via signals/slots, and never touches GUI widgets directly. A `step_completed = pyqtSignal(dict)` signal carries simulation state to the main thread, where a `QTimer` at 20–30ms intervals triggers plot updates. The continuous-loop pattern uses `QTimer.singleShot(0, self.worker.run)` connected to the worker's `end_of_run` signal, creating a non-blocking simulation loop that yields to the event loop between steps.

**pyqtgraph outperforms matplotlib by 25x for real-time rendering** — over 1,000 FPS vs ~40 FPS for a 1,024-point line plot. pyqtgraph renders via Qt's native `QGraphicsScene` (vector-based), while matplotlib renders to bitmaps through the Agg backend. For a simulation dashboard, use `GraphicsLayoutWidget` for multi-plot grids, `PlotWidget` for individual embedded charts, and `PlotDataItem.setData()` for zero-allocation in-place updates. The `pglive` package adds thread-safe live plotting with `DataConnector` objects that can safely receive data from any thread. Reserve matplotlib for static report generation or publication-quality export only.

**Recommended dashboard architecture:**

```
SimulationDashboard (QMainWindow)
├── pyqtgraph DockArea (central widget)
│   ├── Dock: Revenue/profit time series (PlotWidget)
│   ├── Dock: Market share bar chart (BarGraphItem)
│   ├── Dock: Supply chain node graph (QGraphicsScene)
│   └── Dock: Regime state indicator + KPI table
├── Left Dock: Simulation controls (ParameterTree, start/stop/pause)
├── Bottom Dock: Event log (QTextEdit)
└── QThread + SimulationWorker
    ├── Signals: step_data(dict), finished(), error(str)
    └── Continuous loop via QTimer.singleShot(0, run)
```

pyqtgraph's built-in `DockArea` is preferred over Qt's `QDockWidget` for scientific applications — it offers save/restore state serialization, more flexible stacking, and better resize behavior. For the supply chain and business process visualization, **NodeGraphQt** (pip-installable, PySide2/PyQt5, QGraphicsScene-backed) provides a complete node graph framework with custom nodes, ports, connection pipes, undo/redo, and serialization. Alternatively, pyqtgraph's built-in `Flowchart` module supports dataflow visualization integrated directly with pyqtgraph plots.

For shared simulation state between the worker thread and GUI, use either signals carrying immutable data dictionaries (preferred — no locking needed) or a `QMutex`-protected state object when the GUI needs random-access to simulation state. Never access Qt widgets from the worker thread.

---

## 5. Vectorized numpy implementations enable real-time economic computation

All required economic models can be implemented as vectorized numpy operations over N firms simultaneously, making the mathematical computation negligible relative to LLM inference latency. The key pattern: represent all per-firm state as numpy arrays of shape `(N,)` and apply element-wise operations.

**Production functions** compute in O(N) per tick. Cobb-Douglas: `Y = A * np.power(K, alpha) * np.power(L, beta)`. CES requires careful handling of the ρ→0 Cobb-Douglas limit: branch on `abs(rho) < 1e-10`. Optimal input allocation under CES uses analytical solutions derived from first-order conditions rather than numerical optimization, dramatically reducing per-tick compute. Capital evolves via `K_new = K * (1 - depreciation) + investment` each tick.

**Competition models** use iterative best-response. Cournot equilibrium for N firms converges in O(N × iterations) using damped updates: `q_new = λ * BR(q) + (1-λ) * q_old` with λ ∈ (0.3, 0.7) for stability. For weekly ticks, a single best-response step per tick (rather than full convergence) creates realistic adaptive dynamics. Differentiated Bertrand competition uses `demand_i = a - b_own * p_i + b_cross * Σp_j` with profit maximization per firm. For `scipy.optimize.root`, the fixed-point formulation `q - BR(q) = 0` with `method='hybr'` converges reliably.

**Learning curves** create the core growth dynamic. Wright's Law (`cost = a * cumulative_production^(-b)`) updates cumulatively each tick: `cumulative += output; cost = first_unit_cost * np.power(cumulative, -learning_exponent)`. Typical learning rates are **10–25% cost reduction per doubling** (20% for solar, 15–20% for aircraft). This generates positive feedback: lower costs → higher output → more learning → even lower costs.

**Inventory management** uses the (s,S) policy for weekly ticks: if inventory ≤ s (reorder point), order up to S. Safety stock calculation: `z * demand_std * sqrt(lead_time_weeks)` where z = 1.645 for 95% service level. The `stockpyl` library provides production-ready implementations of EOQ, newsvendor, Wagner-Whitin, and multi-echelon optimization.

**Time series generation** combines ARIMA for mean dynamics with GARCH for volatility clustering. Using `statsmodels.tsa.arima_process.arma_generate_sample()` for the demand trend component and a manual GARCH(1,1) loop for heteroskedastic noise produces realistic market demand series. **Markov regime-switching** modulates simulation parameters: a transition matrix (e.g., 95% stay-in-expansion, 90% stay-in-recession) generates regime sequences, and each regime maps to different growth rates, volatility levels, and demand multipliers. The `statsmodels.tsa.regime_switching.MarkovRegression` class handles estimation from historical data; simulation uses simple `np.random.choice(n_regimes, p=transition_matrix[current_state])`.

**Leontief input-output analysis** computes supply chain multiplier effects via `x = (I - A)^(-1) @ d` using `np.linalg.inv` for small matrices (<100 sectors) or `scipy.sparse.linalg.spsolve` for larger systems. This enables modeling how a demand shock in one sector propagates through inter-industry relationships.

---

## 6. SEC EDGAR and edgartools anchor the data ingestion pipeline

**The `edgartools` library** (MIT license, actively maintained) is the recommended primary data source. It wraps SEC EDGAR's free APIs into structured Python objects: `Company("AAPL").get_financials()` returns pandas DataFrames for income statement, balance sheet, and cash flow. No API key required — only a User-Agent header with contact info. The SEC enforces **10 requests/second** rate limits; bulk downloads of all company facts (`companyfacts.zip`, ~5GB) are available for initial database population.

XBRL provides the underlying structure. Key US-GAAP tags map directly to simulation parameters: `Revenues`, `CostOfGoodsAndServicesSold`, `OperatingExpenses`, `PropertyPlantAndEquipmentNet` (capital proxy), `InventoryNet`, `CashAndCashEquivalentsAtCarryingValue`, `LongTermDebt`, and `DepreciationDepletionAndAmortization`. The SEC's **Company Facts API** (`data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`) returns all XBRL data for a company in a single call. The **Frames API** (`data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json`) enables cross-company comparisons for a single metric — essential for parameterizing competitor agents from industry benchmarks.

**Minimum data for firm simulation parameterization** breaks into five categories:

- **Revenue profile:** current annual revenue, 3–5yr CAGR, quarterly seasonality pattern, addressable market share
- **Cost structure:** gross margin, operating expense ratio (SG&A + R&D as % of revenue), fixed/variable cost split, depreciation rate
- **Capital:** total debt, cash position, CapEx intensity (CapEx/Revenue), PP&E/total assets ratio, WACC
- **Working capital:** days sales outstanding, days inventory outstanding, days payable outstanding
- **Production function parameters:** output elasticities of capital (typically α_K ≈ 0.25–0.45) and labor (α_L ≈ 0.55–0.75), estimated from revenue vs. employees and PP&E via Levinsohn-Petrin or Ackerberg-Caves-Frazer methods

The normalization pipeline should convert raw financials to ratio-based parameters (margins, turnover ratios, growth rates) for cross-company comparability, align fiscal years to calendar quarters using SEC Frames API conventions, and output a `SimParams` object that directly initializes a `FirmAgent`. Industry benchmarks by SIC/NAICS code from the Frames API provide defaults for competitor agents when detailed data isn't available.

---

## Conclusion: a three-layer architecture ties everything together

The target application should be structured in **three layers** that separate concerns cleanly. The **Simulation Engine** adopts Mesa's staged-activation pattern with OASIS-inspired `ManualAction`/`LLMAction` duality: a tick loop processes phases (market clearing → firm decisions → production → payments → data collection), where each firm decision routes through the hybrid rule/LLM classifier. The **Economic Core** implements all mathematical models as vectorized numpy operations over firm-state arrays, with Markov regime-switching driving exogenous conditions and Leontief matrices propagating supply chain effects. The **Visualization Layer** uses pyqtgraph's DockArea with a QThread worker pattern, emitting simulation state via signals at each tick for 20–30 FPS dashboard updates.

The most non-obvious architectural insight: **configurable competitor detail levels map naturally to the decision routing tiers**. A "low detail" competitor runs entirely on rule-based heuristics (Tier 1) with statistical parameters drawn from SEC industry benchmarks. A "medium detail" competitor adds GPT-4o-mini for quarterly strategic adjustments. A "high detail" competitor gets full LLM reasoning at every decision point using the user's actual financial data. This design makes computational cost proportional to analytical depth, with a 10-agent, 200-tick simulation costing roughly **$1–3** in API fees rather than $100+ naive. The data pipeline feeds SEC EDGAR financials through `edgartools` → normalization → `SimParams`, enabling users to parameterize simulations from real company data with minimal manual configuration.