"""Save a finished run to disk: full data (JSON), time series (CSV), and the PDF report.

Every run lands in `backend/runs/<run_id>/` so it can be studied or compared
later. The PDF is rendered from the same dict that is written to `run.json`,
so a report can always be regenerated from a saved run.
"""

from __future__ import annotations

import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src.simulation.stance import PROFILES

if TYPE_CHECKING:
    from src.simulation.engine import MarketSim
    from src.simulation.report import CeoReport

RUNS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "runs"
#: Files written per run → media type. Also the whitelist the API serves.
RUN_FILES: Final[dict[str, str]] = {
    "report.pdf": "application/pdf",
    "timeseries.csv": "text/csv",
    "run.json": "application/json",
}
RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9-]{1,120}$")

CSV_FIELDS: Final[tuple[str, ...]] = (
    "day",
    "company_id",
    "company",
    "ceo",
    "alive",
    "cash",
    "revenue",
    "core_revenue",
    "costs",
    "profit",
    "market_share",
    "price",
    "marketing_intensity",
    "satisfaction",
    "utilization",
    "employees",
    "locations",
    "runway_months",
    "binding",
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "run"


def run_data(sim: MarketSim, reports: list[CeoReport]) -> dict[str, Any]:
    by_company = {r.company_id: r for r in reports}
    started = datetime.now()
    seed = sim.companies[0].seed
    return {
        "run_id": f"{started:%Y%m%d-%H%M%S}-{_slug(seed.name)}-{uuid.uuid4().hex[:6]}",
        "created_at": started.isoformat(timespec="seconds"),
        "days": sim.tick,
        "max_days": sim.max_ticks,
        "settings": dict(sim.settings),
        "seed": seed.model_dump(),
        "ai": sim.ai_status(),
        "jev": sim.jev.status() if sim.jev is not None else None,
        "market": {"series": sim.market_series, "events": sim.log},
        "companies": [
            {
                "company_id": c.company_id,
                "name": c.name,
                "archetype": c.stance.archetype,
                "archetype_label": PROFILES[c.stance.archetype]["label"],
                "stance": c.stance.model_dump(),
                "alive": c.alive,
                "died_on": c.died_on,
                "totals": {
                    "revenue": c.total_revenue,
                    "costs": c.total_costs,
                    "profit": c.total_revenue - c.total_costs,
                    "peak_cash": c.peak_cash,
                },
                "series": c.series,
                "decisions": [d.model_dump() for d in c.decisions],
                "events": c.log,
                "report": by_company[c.company_id].model_dump()
                if c.company_id in by_company
                else None,
            }
            for c in sim.companies
        ],
    }


def save_run(data: dict[str, Any], runs_dir: Path | None = None) -> Path:
    """Write run.json, timeseries.csv and report.pdf. Returns the run directory."""
    from src.simulation.report_pdf import render_report

    run_dir = (runs_dir or RUNS_DIR) / data["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run.json").write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    with (run_dir / "timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for company in data["companies"]:
            for row in company["series"]:
                writer.writerow(
                    {
                        **row,
                        "company_id": company["company_id"],
                        "company": company["name"],
                        "ceo": company["archetype"],
                    }
                )
    render_report(data, run_dir / "report.pdf")
    return run_dir


def run_file(run_id: str, filename: str, runs_dir: Path | None = None) -> Path | None:
    """Resolve a saved run file, or None if the id/name is invalid or missing."""
    if not RUN_ID_PATTERN.match(run_id) or filename not in RUN_FILES:
        return None
    path = (runs_dir or RUNS_DIR) / run_id / filename
    return path if path.is_file() else None
