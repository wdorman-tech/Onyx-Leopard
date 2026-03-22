from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class KpiPanel(QWidget):
    """Row of KPI cards, one per active company."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self._cards: dict[str, KpiCard] = {}

    def update_state(self, state: dict) -> None:
        names: list[str] = state.get("company_names", [])
        colors: list[str] = state.get("company_colors", [])
        revenues: list[float] = state.get("revenue", [])
        cash_values: list[float] = state.get("cash", [])
        shares: list[float] = state.get("market_share", [])
        headcounts: list[list[float]] = state.get("dept_headcount", [])
        health_scores: list[float] = state.get("health_score", [])

        for name in list(self._cards.keys()):
            if name not in names:
                card = self._cards.pop(name)
                self._layout.removeWidget(card)
                card.deleteLater()

        for i, name in enumerate(names):
            if name not in self._cards:
                color = colors[i] if i < len(colors) else "#ffffff"
                card = KpiCard(name, color)
                self._cards[name] = card
                self._layout.addWidget(card)

            total_headcount = sum(headcounts[i]) if i < len(headcounts) else 0
            self._cards[name].update_metrics(
                revenue=revenues[i] if i < len(revenues) else 0,
                cash=cash_values[i] if i < len(cash_values) else 0,
                market_share=shares[i] if i < len(shares) else 0,
                headcount=total_headcount,
                health=health_scores[i] if i < len(health_scores) else 1.0,
            )


class KpiCard(QFrame):
    """Single company KPI card."""

    def __init__(self, name: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._base_style = (
            f"QFrame {{ background-color: #0f0f23; border: 2px solid {color}; "
            f"border-radius: 8px; padding: 8px; }}"
        )
        self.setStyleSheet(self._base_style)
        self.setMinimumWidth(150)
        self.setMaximumWidth(220)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self._name_label = QLabel(name)
        self._name_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._name_label)

        self._revenue_label = QLabel("Revenue: $0")
        self._cash_label = QLabel("Cash: $0")
        self._share_label = QLabel("Share: 0%")
        self._headcount_label = QLabel("Headcount: 0")

        for label in [
            self._revenue_label,
            self._cash_label,
            self._share_label,
            self._headcount_label,
        ]:
            label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
            layout.addWidget(label)

        self._prev_revenue: float = 0.0

    def update_metrics(
        self,
        revenue: float,
        cash: float,
        market_share: float,
        headcount: float,
        health: float,
    ) -> None:
        self._revenue_label.setText(f"Revenue: ${revenue:,.0f}")
        self._cash_label.setText(f"Cash: ${cash:,.0f}")
        self._share_label.setText(f"Share: {market_share:.1%}")
        self._headcount_label.setText(f"Headcount: {headcount:.0f}")

        if revenue > self._prev_revenue:
            self._pulse_border("#2ECC71")
        elif revenue < self._prev_revenue:
            self._pulse_border("#E74C3C")
        self._prev_revenue = revenue

    def _pulse_border(self, color: str) -> None:
        """Briefly flash the card border color then revert after 200ms."""
        self.setStyleSheet(
            f"QFrame {{ background-color: #0f0f23; border: 2px solid {color}; "
            f"border-radius: 8px; padding: 8px; }}"
        )
        QTimer.singleShot(200, self._revert_border)

    def _revert_border(self) -> None:
        self.setStyleSheet(self._base_style)
