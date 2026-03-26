"""Fixed color palette for market agent visualization."""

from __future__ import annotations

AGENT_COLORS: list[str] = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#db2777",  # pink
    "#65a30d",  # lime
    "#ea580c",  # orange
    "#4f46e5",  # indigo
    "#0d9488",  # teal
    "#c026d3",  # fuchsia
    "#ca8a04",  # yellow
    "#059669",  # emerald
    "#e11d48",  # rose
    "#2dd4bf",  # teal-light
    "#a855f7",  # purple
    "#f97316",  # orange-light
    "#6366f1",  # indigo-light
    "#14b8a6",  # teal-mid
]


def agent_color(index: int) -> str:
    return AGENT_COLORS[index % len(AGENT_COLORS)]
