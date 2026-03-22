# BioSim

Biological Agent-Based Business Simulation Engine. Companies are living organisms on a petri dish — departments are colored cells that grow, compete, and die based on ODE-driven economics.

**Black Lily LLC** | Phase 1: Minimal Viable Simulation

## Quick Start

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/wdorman-tech/Onyx-Leopard.git
cd Onyx-Leopard
uv sync
uv run python -m biosim
```

Three default companies launch automatically. Press **Play** to start the simulation.

## Controls

- **Play/Pause** — start or pause the tick loop
- **Step** — advance one tick (1 simulated week)
- **Speed slider** — 1x to 20x simulation speed
- **Tab: Petri Dish** — biological visualization with organism blobs and colored department cells
- **Tab: Dashboard** — real-time revenue, market share, and cash charts with KPI cards

## Tech Stack

| Layer | Technology |
|-------|-----------|
| GUI | PyQt6, pyqtgraph |
| Agents | Mesa 3.x |
| Math | numpy, scipy (vectorized ODE solving) |
| State | Struct-of-arrays numpy (column-oriented) |

## Math Models (Phase 1)

- **Logistic growth** — firm size dynamics with carrying capacity
- **Cobb-Douglas production** — Y = A * K^alpha * L^beta
- **Lotka-Volterra competition** — N-species market share dynamics

## Development

```bash
uv sync --all-extras
uv run pytest tests/ -v        # 132 tests
uv run ruff check src/ tests/  # lint
```

## License

MIT
