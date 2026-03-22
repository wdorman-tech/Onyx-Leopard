from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView

from biosim.gui.petri.organism import OrganismItem


class PetriDishView(QGraphicsView):
    """Interactive view with pan (click-drag) and zoom (scroll)."""

    def __init__(self, parent=None) -> None:
        self._scene = PetriDishScene()
        super().__init__(self._scene, parent)
        self.setRenderHint(self.renderHints().Antialiasing, True)
        self.setRenderHint(self.renderHints().SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#1a1a2e")))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    @property
    def petri_scene(self) -> PetriDishScene:
        return self._scene

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom with scroll wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


class PetriDishScene(QGraphicsScene):
    """The petri dish arena containing organism items."""

    DISH_RADIUS = 400.0

    def __init__(self) -> None:
        super().__init__()
        self.setSceneRect(-500, -500, 1000, 1000)
        self._organisms: dict[int, OrganismItem] = {}
        self._draw_dish_border()

    def _draw_dish_border(self) -> None:
        pen = QPen(QColor("#334155"), 3)
        brush = QBrush(Qt.BrushStyle.NoBrush)
        self.addEllipse(
            -self.DISH_RADIUS,
            -self.DISH_RADIUS,
            self.DISH_RADIUS * 2,
            self.DISH_RADIUS * 2,
            pen,
            brush,
        )

    @property
    def organisms(self) -> dict[int, OrganismItem]:
        return self._organisms

    def update_state(self, state: dict) -> None:
        """Update all organisms from a state snapshot dict.

        Expected keys:
        - n_active: int
        - indices: list[int]
        - company_names: list[str]
        - company_colors: list[str]
        - firm_size: list[float]
        - health_score: list[float]
        - revenue: list[float]
        - dept_headcount: list[list[float]]  (n_companies x 12)
        """
        active_indices = set(state.get("indices", []))

        dead = [idx for idx in self._organisms if idx not in active_indices]
        for idx in dead:
            organism = self._organisms.pop(idx)
            self.removeItem(organism)

        n_active = state.get("n_active", len(active_indices))
        positions = _distribute_positions(n_active, self.DISH_RADIUS * 0.5)

        for i, idx in enumerate(state.get("indices", [])):
            pos = positions[i] if i < len(positions) else (0.0, 0.0)

            if idx not in self._organisms:
                names = state.get("company_names", [])
                colors = state.get("company_colors", [])
                name = names[i] if i < len(names) else ""
                color = colors[i] if i < len(colors) else "#888"
                organism = OrganismItem(idx, name, color, pos)
                self._organisms[idx] = organism
                self.addItem(organism)

            organism = self._organisms[idx]
            organism.update_from_state(
                firm_size=state["firm_size"][i],
                health_score=state["health_score"][i],
                revenue=state["revenue"][i],
                dept_headcount=state["dept_headcount"][i],
            )


def _distribute_positions(
    n: int, orbit_radius: float
) -> list[tuple[float, float]]:
    """Distribute n positions evenly around a circle."""
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    return [
        (
            orbit_radius * math.cos(2 * math.pi * i / n),
            orbit_radius * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
