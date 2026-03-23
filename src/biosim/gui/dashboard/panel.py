from __future__ import annotations

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from biosim.gui.dashboard.charts import CashChart, MarketShareChart, RevenueChart
from biosim.gui.dashboard.decision_log import DecisionLogWidget
from biosim.gui.dashboard.kpi import KpiPanel


class DashboardPanel(QWidget):
    """Combined dashboard: KPIs on top, charts below."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.kpi_panel = KpiPanel()
        layout.addWidget(self.kpi_panel)

        self.revenue_chart = RevenueChart()
        layout.addWidget(self.revenue_chart, stretch=1)

        self.market_share_chart = MarketShareChart()
        layout.addWidget(self.market_share_chart, stretch=1)

        self.cash_chart = CashChart()
        layout.addWidget(self.cash_chart, stretch=1)

        self.decision_log = DecisionLogWidget()
        layout.addWidget(self.decision_log, stretch=1)

    def update_state(self, state: dict) -> None:
        """Update all dashboard components from state snapshot."""
        self.kpi_panel.update_state(state)
        self.revenue_chart.update_state(state)
        self.market_share_chart.update_state(state)
        self.cash_chart.update_state(state)
        self.decision_log.update_state(state)

    def clear_data(self) -> None:
        self.revenue_chart.clear_data()
        self.market_share_chart.clear_data()
        self.cash_chart.clear_data()
        self.decision_log.clear_data()
