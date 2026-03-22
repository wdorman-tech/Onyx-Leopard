from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

COMPANY_PALETTE = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E9",
]


class _LineChart(QWidget):
    """Base for line charts that track one value per company over time."""

    MAX_HISTORY = 200

    def __init__(
        self,
        title: str,
        y_label: str,
        state_key: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state_key = state_key

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title=title)
        self.plot_widget.setLabel("left", y_label)
        self.plot_widget.setLabel("bottom", "Tick (weeks)")
        self.plot_widget.addLegend(offset=(10, 10))
        self.plot_widget.setBackground("#1a1a2e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        self._data: dict[str, dict[str, list[float]]] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._tick: int = 0

    def update_state(self, state: dict) -> None:
        """Update chart from state snapshot dict."""
        self._tick += 1
        names: list[str] = state.get("company_names", [])
        values: list[float] = state.get(self._state_key, [])
        colors: list[str] = state.get("company_colors", [])

        for i, name in enumerate(names):
            if name not in self._data:
                self._data[name] = {"ticks": [], "values": []}
                color = colors[i] if i < len(colors) else COMPANY_PALETTE[i % len(COMPANY_PALETTE)]
                pen = pg.mkPen(color=color, width=2)
                self._curves[name] = self.plot_widget.plot([], [], pen=pen, name=name)

            self._data[name]["ticks"].append(self._tick)
            self._data[name]["values"].append(values[i] if i < len(values) else 0)

            if len(self._data[name]["ticks"]) > self.MAX_HISTORY:
                self._data[name]["ticks"] = self._data[name]["ticks"][-self.MAX_HISTORY :]
                self._data[name]["values"] = self._data[name]["values"][-self.MAX_HISTORY :]

            self._curves[name].setData(
                self._data[name]["ticks"],
                self._data[name]["values"],
            )

    def clear_data(self) -> None:
        """Reset all chart data."""
        self._data.clear()
        for curve in self._curves.values():
            self.plot_widget.removeItem(curve)
        self._curves.clear()
        self._tick = 0


class RevenueChart(_LineChart):
    """Revenue by company over time -- overlapping line chart."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            title="Revenue by Company",
            y_label="Revenue ($)",
            state_key="revenue",
            parent=parent,
        )


class CashChart(_LineChart):
    """Cash position by company with insolvency threshold at $0."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            title="Cash Position",
            y_label="Cash ($)",
            state_key="cash",
            parent=parent,
        )
        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen("#E74C3C", width=2, style=pg.QtCore.Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(self._zero_line)

    def clear_data(self) -> None:
        super().clear_data()


class MarketShareChart(QWidget):
    """Market share by company -- stacked 100% area chart."""

    MAX_HISTORY = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title="Market Share (%)")
        self.plot_widget.setLabel("left", "Share (%)")
        self.plot_widget.setLabel("bottom", "Tick (weeks)")
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.setBackground("#1a1a2e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        self._data: dict[str, dict[str, list[float]]] = {}
        self._tick: int = 0

    def update_state(self, state: dict) -> None:
        """Update chart. market_share values should sum to ~1.0."""
        self._tick += 1
        names: list[str] = state.get("company_names", [])
        shares: list[float] = state.get("market_share", [])
        colors: list[str] = state.get("company_colors", [])

        for i, name in enumerate(names):
            if name not in self._data:
                self._data[name] = {"ticks": [], "values": []}
            share_pct = (shares[i] * 100) if i < len(shares) else 0
            self._data[name]["ticks"].append(self._tick)
            self._data[name]["values"].append(share_pct)
            if len(self._data[name]["ticks"]) > self.MAX_HISTORY:
                self._data[name]["ticks"] = self._data[name]["ticks"][-self.MAX_HISTORY :]
                self._data[name]["values"] = self._data[name]["values"][-self.MAX_HISTORY :]

        self._redraw_stacked(names, colors)

    def _redraw_stacked(self, names: list[str], colors: list[str]) -> None:
        """Redraw the stacked fill areas."""
        self.plot_widget.clear()

        if not names or not self._data:
            return

        all_ticks = sorted({t for d in self._data.values() for t in d["ticks"]})
        if not all_ticks:
            return

        ticks_arr = np.array(all_ticks, dtype=np.float64)
        baseline = np.zeros(len(all_ticks))

        for i, name in enumerate(names):
            if name not in self._data:
                continue
            d = self._data[name]
            values = np.interp(ticks_arr, d["ticks"], d["values"], left=0, right=0)
            top = baseline + values
            color = colors[i] if i < len(colors) else COMPANY_PALETTE[i % len(COMPANY_PALETTE)]

            fill = pg.FillBetweenItem(
                pg.PlotDataItem(ticks_arr, baseline),
                pg.PlotDataItem(ticks_arr, top),
                brush=pg.mkBrush(color + "80"),
            )
            self.plot_widget.addItem(fill)
            self.plot_widget.plot(ticks_arr, top, pen=pg.mkPen(color, width=1))
            baseline = top.copy()

    def clear_data(self) -> None:
        self._data.clear()
        self.plot_widget.clear()
        self._tick = 0
