"""Run report PDF in the Onyx Leopard theme.

Page 1: run summary, leaderboard and the cash race. Then a grid of finance
charts for every company, one section per company with its CEO's post-mortem
and a sample of its decisions, and an appendix of market events.

Rendered from the `run.json` dict (see `export.run_data`), never from live
engine objects, so any saved run can be re-rendered.
"""

from __future__ import annotations

import bisect
import io
import statistics
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import escape

import matplotlib
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from reportlab.graphics.shapes import Circle, Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.simulation.report import clip

# ─── Theme (mirrors frontend/src/app/globals.css) ──────────────────────────

INK: Final = "#0f172a"
TEXT: Final = "#334155"
MUTED: Final = "#64748b"
FAINT: Final = "#94a3b8"
RULE: Final = "#e2e8f0"
PANEL: Final = "#f8fafc"
PANEL_2: Final = "#f1f5f9"
ACCENT: Final = "#3b82f6"
POSITIVE: Final = "#16a34a"
NEGATIVE: Final = "#ef4444"
WARNING: Final = "#f59e0b"
#: Same order as `COMPANY_COLORS` in frontend/src/hooks/useSimulation.ts.
COMPANY_COLORS: Final = (
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
    "#0891b2", "#be123c", "#4f46e5", "#65a30d", "#c2410c",
)  # fmt: skip

PAGE_W, PAGE_H = letter
MARGIN_X: Final = 0.6 * inch
CONTENT_W: Final = PAGE_W - 2 * MARGIN_X
CHART_DPI: Final = 200
#: Days of profit averaged in the profit chart, to see the trend through noise.
PROFIT_SMOOTHING_DAYS: Final = 30
#: Runs longer than this plot in years instead of days.
YEAR_AXIS_AFTER_DAYS: Final = 730
DECISIONS_PER_COMPANY: Final = 10
MAX_EVENT_ROWS: Final = 80

_FONT_CHOICES: Final = ("Inter", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")


def _chart_font() -> str:
    for family in _FONT_CHOICES:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


matplotlib.rcParams["font.family"] = _chart_font()

# ─── Text styles ───────────────────────────────────────────────────────────

_BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=9, leading=13, textColor=TEXT)
S: Final = {
    "kicker": ParagraphStyle("kicker", _BASE, fontName="Helvetica-Bold", fontSize=7.5,
                             textColor=ACCENT, leading=10),
    "title": ParagraphStyle("title", _BASE, fontName="Helvetica-Bold", fontSize=24,
                            leading=28, textColor=INK, spaceBefore=4, spaceAfter=4),
    "h2": ParagraphStyle("h2", _BASE, fontName="Helvetica-Bold", fontSize=13, leading=16,
                         textColor=INK, spaceBefore=10, spaceAfter=6),
    "h3": ParagraphStyle("h3", _BASE, fontName="Helvetica-Bold", fontSize=8, leading=11,
                         textColor=MUTED, spaceBefore=8, spaceAfter=3),
    "body": _BASE,
    "small": ParagraphStyle("small", _BASE, fontSize=7.5, leading=10, textColor=MUTED),
    "label": ParagraphStyle("label", _BASE, fontName="Helvetica-Bold", fontSize=6.5,
                            leading=8, textColor=FAINT),
    "value": ParagraphStyle("value", _BASE, fontName="Helvetica-Bold", fontSize=10.5,
                            leading=13, textColor=INK),
    "cell": ParagraphStyle("cell", _BASE, fontSize=7.5, leading=9.5, textColor=TEXT),
    "cell_r": ParagraphStyle("cell_r", _BASE, fontSize=7.5, leading=9.5, textColor=TEXT,
                             alignment=2),
    "quote": ParagraphStyle("quote", _BASE, fontName="Helvetica-Oblique", fontSize=9.5,
                            leading=14, textColor=INK),
}  # fmt: skip


# ─── Formatting ────────────────────────────────────────────────────────────


def money(n: float | None) -> str:
    if n is None:
        return "–"
    sign = "-" if n < 0 else ""
    a = abs(n)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= scale:
            return f"{sign}${a / scale:.1f}{suffix}"
    return f"{sign}${a:,.0f}"


def usd(n: float) -> str:
    """Exact small dollar amounts (API spend is often fractions of a cent)."""
    return f"${n:.4f}" if n < 0.01 else f"${n:.2f}" if n < 1000 else money(n)


def pct(n: float | None) -> str:
    return "–" if n is None else f"{n * 100:.1f}%"


_TIGHT: Final = [
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ("TOPPADDING", (0, 0), (-1, -1), 0),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]


def _dot(color: str, size: float = 6) -> Drawing:
    """A vector circle; font glyphs for dots get substituted by PDF viewers."""
    d = Drawing(size, size)
    d.add(Circle(size / 2, size / 2, size / 2, fillColor=colors.HexColor(color), strokeColor=None))
    return d


def _chip(color: str, text: str, width: float, style: str = "cell") -> Table:
    """Color dot + label, `width` points wide."""
    table = Table([[_dot(color), _p(text, style)]], colWidths=[10, width - 10])
    table.setStyle(TableStyle(_TIGHT))
    return table


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def _t(text: Any) -> str:
    """Escape model- or user-written text for reportlab markup."""
    return escape(str(text))


def _title(s: str) -> str:
    return s.replace("_", " ").title()


# ─── Page furniture ────────────────────────────────────────────────────────


def _draw_logo(canvas: Any, x: float, y: float, size: float) -> None:
    """The app's bolt mark in a rounded tile, bottom-left at (x, y)."""
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(PANEL_2))
    canvas.setStrokeColor(colors.HexColor(RULE))
    canvas.setLineWidth(0.6)
    canvas.roundRect(x, y, size, size, size * 0.22, stroke=1, fill=1)
    # SVG path "M13 2L3 14h9l-1 8 10-12h-9l1-8z" on a 24-unit grid, y flipped.
    unit = size * 0.62 / 24
    ox, oy = x + size * 0.19, y + size * 0.81
    points = [(13, 2), (3, 14), (12, 14), (11, 22), (21, 10), (12, 10)]
    path = canvas.beginPath()
    path.moveTo(ox + points[0][0] * unit, oy - points[0][1] * unit)
    for px, py in points[1:]:
        path.lineTo(ox + px * unit, oy - py * unit)
    path.close()
    canvas.setStrokeColor(colors.HexColor(ACCENT))
    canvas.setLineWidth(2 * unit)
    canvas.setLineJoin(1)
    canvas.drawPath(path, stroke=1, fill=0)
    canvas.restoreState()


def _page(run: dict[str, Any]) -> Callable[[Any, Any], None]:
    name = run["seed"]["name"]

    def draw(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        top = PAGE_H - 0.5 * inch
        _draw_logo(canvas, MARGIN_X, top - 14, 16)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor(INK))
        canvas.drawString(MARGIN_X + 22, top - 9.5, "ONYX LEOPARD")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawRightString(PAGE_W - MARGIN_X, top - 9.5, f"Run report · {name}")
        canvas.setStrokeColor(colors.HexColor(RULE))
        canvas.setLineWidth(0.6)
        canvas.line(MARGIN_X, top - 22, PAGE_W - MARGIN_X, top - 22)
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(colors.HexColor(FAINT))
        canvas.drawString(MARGIN_X, 0.45 * inch, run["run_id"])
        canvas.drawRightString(PAGE_W - MARGIN_X, 0.45 * inch, f"Page {doc.page}")
        canvas.restoreState()

    return draw


# ─── Charts ────────────────────────────────────────────────────────────────


def _axes(width_in: float, height_in: float, title: str) -> tuple[Figure, Any]:
    fig = Figure(figsize=(width_in, height_in), dpi=CHART_DPI, facecolor="white")
    ax = fig.add_axes((0.12, 0.16, 0.85, 0.66))
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", color=INK, pad=7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=6, length=0, pad=3)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    return fig, ax


def _image(fig: Figure, width_in: float, height_in: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI, facecolor="white")
    buf.seek(0)
    return Image(buf, width=width_in * inch, height=height_in * inch)


class _Axis:
    """Day or year x-axis, chosen once per run so every chart agrees."""

    def __init__(self, days: int) -> None:
        self.years = days > YEAR_AXIS_AFTER_DAYS
        self.label = "Year" if self.years else "Day"

    def x(self, day: float) -> float:
        return day / 365 if self.years else day

    def style(self, ax: Any, days: int) -> None:
        ax.set_xlim(0, self.x(days))
        ax.set_xlabel(self.label, fontsize=6, color=FAINT, labelpad=2)


def _shade_shocks(ax: Any, axis: _Axis, events: list[dict[str, Any]]) -> None:
    for e in events:
        if e.get("kind") == "shock":
            start = axis.x(e["day"])
            ax.axvspan(
                start,
                axis.x(e["day"] + e["duration_days"]),
                color=WARNING,
                alpha=0.08 if e["severity"] != "severe" else 0.16,
                linewidth=0,
            )


def _series(company: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    rows = company["series"]
    return [r["day"] for r in rows], [r[key] if r[key] is not None else 0.0 for r in rows]


def _smooth(days: list[float], values: list[float], window_days: int) -> list[float]:
    if len(days) < 2:
        return values
    step = statistics.median(b - a for a, b in pairwise(days)) or 1
    window = max(1, round(window_days / step))
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def _cumulative(days: list[float], values: list[float]) -> list[float]:
    out, total, prev = [], 0.0, 0.0
    for d, v in zip(days, values, strict=True):
        total += v * (d - prev)
        prev = d
        out.append(total)
    return out


_FORMATTERS: Final = {
    "money": FuncFormatter(lambda v, _: money(v)),
    "pct": FuncFormatter(lambda v, _: f"{v * 100:.0f}%"),
    "int": FuncFormatter(lambda v, _: f"{v:,.0f}"),
    "x": FuncFormatter(lambda v, _: f"{v:.2f}×"),
    "price": FuncFormatter(lambda v, _: f"${v:,.2f}" if abs(v) < 100 else money(v)),
}


def _metric_chart(
    run: dict[str, Any],
    axis: _Axis,
    title: str,
    key: str,
    fmt: str,
    width_in: float,
    height_in: float,
    transform: str | None = None,
) -> Image:
    fig, ax = _axes(width_in, height_in, title)
    _shade_shocks(ax, axis, run["market"]["events"])
    for i, c in enumerate(run["companies"]):
        days, values = _series(c, key)
        if not days:
            continue
        if transform == "smooth":
            values = _smooth(days, values, PROFIT_SMOOTHING_DAYS)
        elif transform == "cumulative":
            values = _cumulative(days, values)
        color = COMPANY_COLORS[i % len(COMPANY_COLORS)]
        xs = [axis.x(d) for d in days]
        drawstyle = "steps-post" if key == "locations" else "default"
        ax.plot(xs, values, color=color, linewidth=1.3, drawstyle=drawstyle)
        if c["died_on"] is not None:
            ax.plot(xs[-1], values[-1], marker="x", color=color, markersize=5, mew=1.5)
    if transform in ("smooth", "cumulative") or key == "cash":
        ax.axhline(0, color=FAINT, linewidth=0.6)
    ax.yaxis.set_major_formatter(_FORMATTERS[fmt])
    if fmt == "int":
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    axis.style(ax, run["days"])
    return _image(fig, width_in, height_in)


def _share_chart(run: dict[str, Any], axis: _Axis, width_in: float, height_in: float) -> Image:
    fig, ax = _axes(width_in, height_in, "Market share")
    days = sorted({r["day"] for c in run["companies"] for r in c["series"]})
    stacks, palette = [], []
    for i, c in enumerate(run["companies"]):
        by_day = {r["day"]: r["market_share"] for r in c["series"]}
        stacks.append([by_day.get(d, 0.0) for d in days])
        palette.append(COMPANY_COLORS[i % len(COMPANY_COLORS)])
    if days:
        ax.stackplot([axis.x(d) for d in days], stacks, colors=palette, alpha=0.85, linewidth=0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(_FORMATTERS["pct"])
    axis.style(ax, run["days"])
    return _image(fig, width_in, height_in)


def _tam_chart(run: dict[str, Any], axis: _Axis, width_in: float, height_in: float) -> Image:
    fig, ax = _axes(width_in, height_in, "Market size (annual TAM)")
    _shade_shocks(ax, axis, run["market"]["events"])
    rows = run["market"]["series"]
    ax.plot([axis.x(r["day"]) for r in rows], [r["tam"] for r in rows], color=INK, linewidth=1.3)
    ax.yaxis.set_major_formatter(_FORMATTERS["money"])
    axis.style(ax, run["days"])
    return _image(fig, width_in, height_in)


_MARKERS: Final = {
    "ai": ("o", ACCENT, "AI CEO"),
    "heuristic": ("^", FAINT, "Rule"),
    "jev_ok": ("D", POSITIVE, "Jev approved"),
    "jev_veto": ("x", NEGATIVE, "Jev vetoed"),
}


def _decision_kind(d: dict[str, Any]) -> str:
    if d["source"] == "jev":
        return "jev_ok" if d.get("jev_approved") else "jev_veto"
    return d["source"]


def _company_chart(run: dict[str, Any], axis: _Axis, company: dict[str, Any], color: str) -> Image:
    width_in, height_in = CONTENT_W / inch, 2.2
    fig, ax = _axes(width_in, height_in, "Cash, with every CEO decision")
    fig.axes[0].set_position((0.07, 0.16, 0.91, 0.66))
    _shade_shocks(ax, axis, run["market"]["events"] + company["events"])
    days, cash = _series(company, "cash")
    xs = [axis.x(d) for d in days]
    ax.plot(xs, cash, color=color, linewidth=1.5)
    ax.fill_between(xs, cash, 0, color=color, alpha=0.06, linewidth=0)
    ax.axhline(0, color=FAINT, linewidth=0.6)
    seen = set()
    for d in company["decisions"]:
        kind = _decision_kind(d)
        marker, mcolor, label = _MARKERS[kind]
        y = _interp(days, cash, d["day"])
        ax.plot(
            axis.x(d["day"]),
            y,
            marker=marker,
            color=mcolor,
            markersize=3.5,
            linestyle="none",
            mew=1.2,
            label=None if kind in seen else label,
        )
        seen.add(kind)
    if seen:
        ax.legend(loc="upper left", fontsize=5.5, frameon=False, ncol=4, handletextpad=0.3)
    ax.yaxis.set_major_formatter(_FORMATTERS["money"])
    axis.style(ax, run["days"])
    return _image(fig, width_in, height_in)


def _interp(days: list[float], values: list[float], day: float) -> float:
    """Linear interpolation of a sampled series at `day` (clamped to its ends)."""
    if not days:
        return 0.0
    i = bisect.bisect_left(days, day)
    if i == 0:
        return values[0]
    if i >= len(days):
        return values[-1]
    d0, d1, v0, v1 = days[i - 1], days[i], values[i - 1], values[i]
    return v0 + (v1 - v0) * (day - d0) / (d1 - d0)


# ─── Tables ────────────────────────────────────────────────────────────────


def _final(company: dict[str, Any]) -> dict[str, Any]:
    return company["series"][-1] if company["series"] else {}


def ranked(companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Survivors by final cash, then the bankrupt by how long they lasted."""
    return sorted(
        companies,
        key=lambda c: (
            c["alive"],
            _final(c).get("cash", 0.0) if c["alive"] else (c["died_on"] or 0),
        ),
        reverse=True,
    )


def _grid_style(header: bool = True) -> list[tuple]:
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(RULE))]
    return style


def _meta_grid(items: list[tuple[str, str]], cols: int = 4) -> Table:
    cells = [[_p(_t(label).upper(), "label"), _p(value, "value")] for label, value in items]
    rows = [cells[i : i + cols] for i in range(0, len(cells), cols)]
    rows[-1] += [""] * (cols - len(rows[-1]))
    table = Table([[_stack(c) for c in row] for row in rows], colWidths=[CONTENT_W / cols] * cols)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                ("GRID", (0, 0), (-1, -1), 4, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def _stack(cell: Any) -> Any:
    return cell if cell == "" else [cell[0], Spacer(1, 2), cell[1]]


#: Leaderboard column widths in inches (sums to the content width).
LEADERBOARD_WIDTHS: Final = (0.24, 1.3, 0.95, 0.88, 0.67, 0.67, 0.67, 0.67, 0.45, 0.42, 0.38)


def _leaderboard(run: dict[str, Any]) -> Table:
    colors_by_id = _color_map(run)
    header = [
        "#",
        "Company",
        "CEO",
        "Outcome",
        "Final cash",
        "Revenue",
        "Profit",
        "Peak cash",
        "Share",
        "Staff",
        "Locs",
    ]
    rows: list[list[Any]] = [[_p(h.upper(), "label") for h in header]]
    for rank, c in enumerate(ranked(run["companies"]), 1):
        final, totals = _final(c), c["totals"]
        outcome = (
            f'<font color="{POSITIVE}">Alive</font>'
            if c["alive"]
            else f'<font color="{NEGATIVE}">Bankrupt · day {c["died_on"]}</font>'
        )
        name = f"<b>{_t(c['name'])}</b>" if rank == 1 else _t(c["name"])
        rows.append(
            [
                _p(str(rank), "cell"),
                _chip(colors_by_id[c["company_id"]], name, LEADERBOARD_WIDTHS[1] * inch - 8),
                _p(_t(c["archetype_label"]), "cell"),
                _p(outcome, "cell"),
                *[
                    _p(v, "cell_r")
                    for v in (
                        money(final.get("cash")),
                        money(totals["revenue"]),
                        money(totals["profit"]),
                        money(totals["peak_cash"]),
                        pct(final.get("market_share")),
                        f"{final.get('employees', 0):,}",
                        str(final.get("locations", 0)),
                    )
                ],
            ]
        )
    table = Table(rows, colWidths=[w * inch for w in LEADERBOARD_WIDTHS], repeatRows=1)
    style = _grid_style()
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(PANEL)))
    table.setStyle(TableStyle(style))
    return table


def _color_map(run: dict[str, Any]) -> dict[str, str]:
    return {
        c["company_id"]: COMPANY_COLORS[i % len(COMPANY_COLORS)]
        for i, c in enumerate(run["companies"])
    }


def _legend(run: dict[str, Any]) -> Table:
    cols = min(5, len(run["companies"]))
    chips = [
        _chip(COMPANY_COLORS[i % len(COMPANY_COLORS)], _t(c["name"]), CONTENT_W / cols - 6, "small")
        for i, c in enumerate(run["companies"])
    ]
    rows = [chips[i : i + cols] for i in range(0, len(chips), cols)]
    rows[-1] += [""] * (cols - len(rows[-1]))
    table = Table(rows, colWidths=[CONTENT_W / cols] * cols)
    table.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return table


# ─── Sections ──────────────────────────────────────────────────────────────


def _years(days: int) -> str:
    return f"{days / 365:.1f} years" if days >= 365 else f"{days} days"


def _summary_section(run: dict[str, Any], axis: _Axis) -> list[Any]:
    seed, ai, jev, settings = run["seed"], run["ai"], run["jev"], run["settings"]
    companies = run["companies"]
    survivors = sum(c["alive"] for c in companies)
    shocks = [e for e in run["market"]["events"] if e.get("kind") == "shock"]
    shocks += [e for c in companies for e in c["events"] if e.get("kind") == "shock"]
    tam = run["market"]["series"]
    ai_value = (
        f"{usd(ai['spent_usd'])} of {usd(ai['budget_usd'])} · {ai['calls']} calls" if ai else "Off"
    )
    jev_value = (
        f"{jev['approved']} approved · {jev['vetoed']} vetoed" if jev else "Off (rules alone)"
    )
    items = [
        ("Duration", _years(run["days"])),
        ("Companies", f"{len(companies)} · {survivors} survived"),
        ("CEO model", _t(ai["model"]) if ai else "Heuristic only"),
        ("AI spend", ai_value),
        ("Jev weekly judge", jev_value),
        ("Market shocks", f"{len(shocks)}" if settings.get("shocks_enabled", True) else "Off"),
        ("Starting cash", money(seed["starting_cash"])),
        ("Annual TAM", f"{money(tam[0]['tam'])} → {money(tam[-1]['tam'])}" if tam else "–"),
    ]

    order = ranked(companies)
    winner = order[0]
    lines = [
        f"<b>{_t(winner['name'])}</b> ({_t(winner['archetype_label'])}) "
        + (
            f"finished on top with {money(_final(winner).get('cash'))} cash and "
            f"{pct(_final(winner).get('market_share'))} of the market."
            if winner["alive"]
            else f"lasted longest, to day {winner['died_on']}."
        )
    ]
    dead = [c for c in companies if not c["alive"]]
    if dead:
        lines.append(
            "Bankrupt: " + ", ".join(f"{_t(c['name'])} (day {c['died_on']})" for c in dead) + "."
        )

    return [
        Spacer(1, 6),
        _p("SIMULATION RUN REPORT", "kicker"),
        _p(_t(seed["name"]), "title"),
        _p(_t(seed["niche"]), "body"),
        _p(
            f"{_t(run['created_at'].replace('T', ' '))} · {_t(seed['economics_model'])} economics",
            "small",
        ),
        Spacer(1, 12),
        _meta_grid(items),
        _p("Leaderboard", "h2"),
        _leaderboard(run),
        Spacer(1, 6),
        _p(" ".join(lines), "body"),
        Spacer(1, 8),
        _metric_chart(run, axis, "Cash over the run", "cash", "money", CONTENT_W / inch, 2.5),
        _legend(run),
    ]


def _charts_section(run: dict[str, Any], axis: _Axis) -> list[Any]:
    w, h = CONTENT_W / inch / 2, 2.0
    charts = [
        _metric_chart(run, axis, "Revenue per day", "revenue", "money", w, h),
        _metric_chart(
            run,
            axis,
            f"Profit per day · {PROFIT_SMOOTHING_DAYS}-day average",
            "profit",
            "money",
            w,
            h,
            "smooth",
        ),
        _metric_chart(run, axis, "Cumulative profit", "profit", "money", w, h, "cumulative"),
        _share_chart(run, axis, w, h),
        _metric_chart(run, axis, "Unit price", "price", "price", w, h),
        _metric_chart(run, axis, "Marketing intensity", "marketing_intensity", "x", w, h),
        _metric_chart(run, axis, "Headcount", "employees", "int", w, h),
        _metric_chart(run, axis, "Locations", "locations", "int", w, h),
        _metric_chart(run, axis, "Customer satisfaction", "satisfaction", "pct", w, h),
        _metric_chart(run, axis, "Capacity utilization", "utilization", "pct", w, h),
        _tam_chart(run, axis, w, h),
    ]
    rows = [charts[i : i + 2] for i in range(0, len(charts), 2)]
    rows[-1] += [""] * (2 - len(rows[-1]))
    grid = Table(rows, colWidths=[CONTENT_W / 2] * 2)
    grid.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return [
        PageBreak(),
        _p("Finances", "h2"),
        _p(
            "Every company, sampled across the whole run. Shaded bands are market shocks "
            "(darker = severe); × marks a bankruptcy.",
            "small",
        ),
        Spacer(1, 4),
        _legend(run),
        grid,
    ]


def _decision_actions(d: dict[str, Any]) -> str:
    parts = [f"+{k}" for k in d["spawn"]] + [f"−{k}" for k in d["retire"]]
    if d["open_locations"]:
        parts.append(f"open {d['open_locations']} location(s)")
    if d["price"] is not None:
        parts.append(f"price ${d['price']:,.2f}")
    if d["marketing_intensity"] is not None:
        parts.append(f"marketing {d['marketing_intensity']:.2f}×")
    if d["raise_amount"]:
        parts.append(f"raise {money(d['raise_amount'])}")
    return ", ".join(parts) or "hold"


def _sample_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spread picks evenly across the run, preferring AI and Jev calls."""
    pool = [d for d in decisions if d["source"] != "heuristic"] or decisions
    if len(pool) <= DECISIONS_PER_COMPANY:
        return pool
    step = (len(pool) - 1) / (DECISIONS_PER_COMPANY - 1)
    return [pool[round(i * step)] for i in range(DECISIONS_PER_COMPANY)]


_SOURCE_LABEL: Final = {
    "ai": f'<font color="{ACCENT}">AI</font>',
    "heuristic": f'<font color="{MUTED}">Rule</font>',
    "jev_ok": f'<font color="{POSITIVE}">Jev ok</font>',
    "jev_veto": f'<font color="{NEGATIVE}">Jev veto</font>',
}


def _company_section(run: dict[str, Any], axis: _Axis, c: dict[str, Any], color: str) -> list:
    stance, totals, final = c["stance"], c["totals"], _final(c)
    decisions = c["decisions"]
    counts = {k: sum(_decision_kind(d) == k for d in decisions) for k in _MARKERS}
    status = (
        f'<font color="{POSITIVE}">Alive</font>'
        if c["alive"]
        else f'<font color="{NEGATIVE}">Bankrupt on day {c["died_on"]}</font>'
    )
    header = Table(
        [
            [
                "",
                [
                    _p(_t(c["name"]), "h2"),
                    _p(f"{_t(c['archetype_label'])} CEO · {status}", "small"),
                ],
            ]
        ],
        colWidths=[5, CONTENT_W - 5],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(color)),
                ("LEFTPADDING", (1, 0), (1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    persona = (
        f"Risk {stance['risk_tolerance']:.2f} · growth {stance['growth_obsession']:.2f} · "
        f"quality {stance['quality_floor']:.2f} · {stance['hiring_bias'].replace('_', ' ')} "
        f"hiring · {stance['time_horizon']} horizon · nervous below "
        f"{stance['cash_comfort']:.1f} months of runway. Signature moves: "
        + "; ".join(f"<i>{_t(m)}</i>" for m in stance["signature_moves"])
        + "."
    )
    stats = _meta_grid(
        [
            ("Final cash", money(final.get("cash"))),
            ("Total revenue", money(totals["revenue"])),
            ("Total profit", money(totals["profit"])),
            ("Peak cash", money(totals["peak_cash"])),
            ("Final share", pct(final.get("market_share"))),
            ("Staff · locations", f"{final.get('employees', 0):,} · {final.get('locations', 0)}"),
            ("Decisions", f"{counts['ai']} AI · {counts['heuristic']} rule"),
            ("Jev", f"{counts['jev_ok']} ok · {counts['jev_veto']} veto"),
        ]
    )

    flow: list[Any] = [
        CondPageBreak(4.5 * inch),
        KeepTogether(
            [
                Spacer(1, 6),
                header,
                _p(persona, "small"),
                Spacer(1, 6),
                stats,
                Spacer(1, 4),
                _company_chart(run, axis, c, color),
            ]
        ),
    ]

    report = c.get("report")
    if report:
        by = "In the CEO's own words" if report["source"] == "ai" else "Heuristic summary"
        well_wrong = Table(
            [
                [
                    _p(f'<font color="{POSITIVE}">WHAT WENT WELL</font>', "label"),
                    _p(f'<font color="{NEGATIVE}">WHAT WENT WRONG</font>', "label"),
                ],
                [_p(_t(report["what_went_well"])), _p(_t(report["what_went_wrong"]))],
            ],
            colWidths=[CONTENT_W / 2] * 2,
        )
        well_wrong.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        verdict = Table(
            [["", _p(_t(report["final_assessment"]), "quote")]], colWidths=[3, CONTENT_W - 3]
        )
        verdict.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(ACCENT)),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(PANEL)),
                    ("LEFTPADDING", (1, 0), (1, 0), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        flow += [
            _p(f"CEO POST-MORTEM · {by.upper()}", "h3"),
            _p(_t(report["performance_summary"])),
            Spacer(1, 6),
            well_wrong,
        ]
        if report["key_decisions"]:
            bullets = [_p(f"&bull; {_t(k)}", "body") for k in report["key_decisions"]]
            flow.append(KeepTogether([_p("KEY DECISIONS", "h3"), *bullets]))
        flow += [Spacer(1, 6), KeepTogether([_p("FINAL ASSESSMENT", "h3"), verdict])]

    sample = _sample_decisions(decisions)
    if sample:
        rows: list[list[Any]] = [[_p(h, "label") for h in ("DAY", "BY", "ACTION", "REASONING")]]
        for d in sample:
            reasoning = clip(d["reasoning"], 260)
            rows.append(
                [
                    _p(str(d["day"]), "cell"),
                    _p(_SOURCE_LABEL[_decision_kind(d)], "cell"),
                    _p(_t(_decision_actions(d)), "cell"),
                    _p(_t(reasoning), "cell"),
                ]
            )
        table = Table(
            rows,
            colWidths=[0.4 * inch, 0.55 * inch, 1.75 * inch, CONTENT_W - 2.7 * inch],
            repeatRows=1,
        )
        style = [*_grid_style(), ("VALIGN", (0, 0), (-1, -1), "TOP")]
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(PANEL)))
        table.setStyle(TableStyle(style))
        flow.append(
            KeepTogether([_p(f"DECISION LOG · {len(sample)} OF {len(decisions)}", "h3"), table])
        )
    return flow


def _events_section(run: dict[str, Any]) -> list[Any]:
    names = {c["company_id"]: c["name"] for c in run["companies"]}
    events = list(run["market"]["events"])
    for c in run["companies"]:
        events += c["events"]
    events.sort(key=lambda e: e["day"])
    if not events:
        return []
    rows: list[list[Any]] = [[_p(h, "label") for h in ("DAY", "WHO", "EVENT", "SEVERITY")]]
    for e in events[:MAX_EVENT_ROWS]:
        who = "Market" if e.get("company_id") is None else names.get(e["company_id"], "")
        if e["kind"] == "bankruptcy":
            what, severity = f'<font color="{NEGATIVE}">Went bankrupt</font>', ""
        else:
            what = f"{_t(_title(e['name']))} · {e['duration_days']} days"
            severity = _t(e["severity"])
        rows.append(
            [_p(str(e["day"]), "cell"), _p(_t(who), "cell"), _p(what, "cell"), _p(severity, "cell")]
        )
    table = Table(
        rows, colWidths=[0.6 * inch, 1.8 * inch, CONTENT_W - 3.3 * inch, 0.9 * inch], repeatRows=1
    )
    style = _grid_style()
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(PANEL)))
    table.setStyle(TableStyle(style))
    flow: list[Any] = [CondPageBreak(2 * inch), _p("Market events", "h2"), table]
    if len(events) > MAX_EVENT_ROWS:
        flow.append(_p(f"{len(events) - MAX_EVENT_ROWS} more in run.json.", "small"))
    return flow


def render_report(run: dict[str, Any], path: Path) -> None:
    axis = _Axis(run["days"])
    doc = BaseDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=0.95 * inch,
        bottomMargin=0.75 * inch,
        title=f"Onyx Leopard run report · {run['seed']['name']}",
        author="Onyx Leopard",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=_page(run))])

    story: list[Any] = _summary_section(run, axis) + _charts_section(run, axis)
    story += [PageBreak(), _p("Companies", "h2")]
    colors_by_id = _color_map(run)
    for c in ranked(run["companies"]):
        story += _company_section(run, axis, c, colors_by_id[c["company_id"]])
    story += _events_section(run)
    doc.build(story)
