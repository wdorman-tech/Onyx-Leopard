from __future__ import annotations

import asyncio
import csv
import json

import pytest

from src.main import StartRequest, build_sim
from src.simulation import export
from src.simulation.engine import MAX_SERIES_POINTS, MarketSim
from src.simulation.report import heuristic_report


def finished_sim(days: int, **kw) -> MarketSim:  # type: ignore[no-untyped-def]
    sim = asyncio.run(
        build_sim(
            StartRequest(
                ceo_archetypes=kw.pop("ceos", ["bootstrap", "venture_growth", "turnaround"]),
                duration_days=days,
                ai_enabled=False,
                jev_enabled=False,
                rng_seed=kw.pop("rng_seed", 5),
                **kw,
            )
        )
    )

    async def go() -> None:
        while not sim.is_complete:
            await sim.step()

    asyncio.run(go())
    return sim


def test_series_covers_whole_run_with_bounded_samples() -> None:
    short = finished_sim(200)
    for c in short.companies:
        assert [r["day"] for r in c.series] == list(range(1, 201))  # daily up to the cap
    long = finished_sim(5 * 365, ceos=["bootstrap"])
    rows = long.companies[0].series
    assert len(rows) <= MAX_SERIES_POINTS + 1
    assert rows[0]["day"] == long.sample_every and rows[-1]["day"] == 5 * 365


def test_totals_match_the_daily_books() -> None:
    sim = finished_sim(120, ceos=["bootstrap"])
    c = sim.companies[0]
    assert c.total_revenue == pytest.approx(sum(r["revenue"] for r in c.series))
    assert c.peak_cash >= max(r["cash"] for r in c.series)


def test_bankruptcy_is_recorded_with_its_day() -> None:
    sim = asyncio.run(
        build_sim(
            StartRequest(
                ceo_archetypes=["bootstrap", "turnaround"],
                duration_days=120,
                ai_enabled=False,
                jev_enabled=False,
                shocks_enabled=False,
            )
        )
    )
    doomed = sim.companies[0]

    async def go() -> None:
        while not sim.is_complete:
            if doomed.alive:
                doomed.cash = -1e12
            await sim.step()

    asyncio.run(go())
    assert doomed.died_on is not None
    assert doomed.series[-1]["day"] == doomed.died_on
    assert doomed.series[-1]["alive"] is False
    assert {"day": doomed.died_on, "kind": "bankruptcy", "company_id": "co-0"} in doomed.log


def test_save_run_writes_json_csv_and_pdf(tmp_path) -> None:  # type: ignore[no-untyped-def]
    sim = finished_sim(2 * 365)
    data = export.run_data(sim, [heuristic_report(c) for c in sim.companies])
    run_dir = export.save_run(data, tmp_path)

    assert sorted(p.name for p in run_dir.iterdir()) == sorted(export.RUN_FILES)
    pdf = (run_dir / "report.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and len(pdf) > 50_000
    saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == run_dir.name
    assert saved["settings"]["jev_enabled"] is False
    with (run_dir / "timeseries.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(len(c.series) for c in sim.companies)
    assert set(rows[0]) == set(export.CSV_FIELDS)

    assert export.run_file(run_dir.name, "report.pdf", tmp_path) == run_dir / "report.pdf"
    assert export.run_file("../etc", "report.pdf", tmp_path) is None
    assert export.run_file(run_dir.name, "secrets.env", tmp_path) is None
