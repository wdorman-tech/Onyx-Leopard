from __future__ import annotations

from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from biosim.types.enums import Department

DEPARTMENT_COLORS = [d.hex_color for d in Department]
DEPARTMENT_NAMES = [d.label for d in Department]
TIER_LABELS = {0: "ODE", 1: "Heuristic", 2: "Haiku", 3: "Sonnet"}


class DecisionLogWidget(QWidget):
    """Scrolling table of recent agent decisions with department color coding."""

    MAX_ROWS = 100

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Agent Decision Log")
        title.setStyleSheet(
            "color: #e0e0e0; font-weight: bold; font-size: 13px; padding: 4px;"
        )
        layout.addWidget(title)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Tick", "Company", "Department", "Tier", "Action", "Confidence"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setStyleSheet(
            "QTableWidget { background: #0f0f23; color: #e0e0e0; "
            "gridline-color: #334155; }"
            "QHeaderView::section { background: #1a1a2e; color: #e0e0e0; "
            "border: 1px solid #334155; }"
        )
        layout.addWidget(self._table)

    def update_state(self, state: dict) -> None:
        """Add new decisions from state snapshot to the log.

        Reads state.get("decisions", []) -- returns empty list when
        the agent pipeline isn't wired up yet, so this widget is safe
        to use before the AI modules are integrated.
        """
        decisions = state.get("decisions", [])
        if not decisions:
            return

        for decision in decisions:
            self._add_row(decision)

        while self._table.rowCount() > self.MAX_ROWS:
            self._table.removeRow(0)

        self._table.scrollToBottom()

    def _add_row(self, decision: dict) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        tick = decision.get("tick", 0)
        company = decision.get("company_name", "")
        dept_idx = decision.get("dept", 0)
        n_depts = len(DEPARTMENT_NAMES)
        dept_name = DEPARTMENT_NAMES[dept_idx % n_depts]
        dept_color = DEPARTMENT_COLORS[dept_idx % n_depts]
        tier = TIER_LABELS.get(decision.get("tier", 0), "?")
        action = decision.get("action", "")
        confidence = decision.get("confidence", 0.0)

        items = [str(tick), company, dept_name, tier, action, f"{confidence:.0%}"]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            if col == 2:
                item.setForeground(QBrush(QColor(dept_color)))
            self._table.setItem(row, col, item)

    def clear_data(self) -> None:
        """Remove all rows from the table."""
        self._table.setRowCount(0)
