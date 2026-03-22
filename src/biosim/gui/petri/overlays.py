from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPen, QRadialGradient
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsScene, QGraphicsTextItem


class TickCounter(QGraphicsTextItem):
    """Displays the current simulation tick in the top-right corner."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tick = 0
        font = QFont("Monospace", 12)
        font.setBold(True)
        self.setFont(font)
        self.setDefaultTextColor(QColor("#aaaaaa"))
        self.setPos(350, -480)
        self._refresh()

    def set_tick(self, tick: int) -> None:
        self._tick = tick
        self._refresh()

    def _refresh(self) -> None:
        self.setPlainText(f"Tick: {self._tick}")


class NutrientGradient(QGraphicsEllipseItem):
    """Semi-transparent radial gradient overlay hinting at market opportunity density."""

    def __init__(self, radius: float = 400.0, parent=None) -> None:
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        gradient = QRadialGradient(QPointF(0, 0), radius)
        center_color = QColor(255, 180, 50, 25)
        edge_color = QColor(255, 180, 50, 0)
        gradient.setColorAt(0, center_color)
        gradient.setColorAt(1, edge_color)
        self.setBrush(QBrush(gradient))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(-10)


def add_overlays(scene: QGraphicsScene, dish_radius: float = 400.0) -> TickCounter:
    """Add standard overlay elements to a petri dish scene. Returns the tick counter."""
    nutrient = NutrientGradient(dish_radius)
    scene.addItem(nutrient)

    tick_counter = TickCounter()
    scene.addItem(tick_counter)

    return tick_counter
