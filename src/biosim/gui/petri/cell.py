from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsEllipseItem

DEPARTMENT_COLORS = [
    "#E74C3C",  # Finance
    "#3498DB",  # R&D
    "#2ECC71",  # Distribution
    "#F1C40F",  # Production
    "#E67E22",  # Sales
    "#9B59B6",  # Marketing
    "#1ABC9C",  # HR
    "#2C3E50",  # Executive
    "#E91E63",  # Customer Service
    "#795548",  # Legal
    "#607D8B",  # IT
    "#8BC34A",  # Procurement
]

DEPARTMENT_NAMES = [
    "Finance",
    "R&D",
    "Distribution",
    "Production",
    "Sales",
    "Marketing",
    "HR",
    "Executive",
    "Customer Service",
    "Legal",
    "IT",
    "Procurement",
]


class CellItem(QGraphicsEllipseItem):
    """A single department cell -- a colored circle positioned by cluster physics."""

    RADIUS = 4.0

    def __init__(
        self, dept_index: int, organism_radius: float, *, seed_spawn: bool = True
    ) -> None:
        super().__init__(-self.RADIUS, -self.RADIUS, self.RADIUS * 2, self.RADIUS * 2)
        self.dept_index = dept_index
        self.update_color()
        self.setPen(QPen(Qt.PenStyle.NoPen))
        rng = np.random.default_rng()
        spawn_radius = organism_radius * (0.15 if seed_spawn else 0.7)
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, spawn_radius)
        self.setPos(dist * np.cos(angle), dist * np.sin(angle))

    def update_color(self) -> None:
        color = QColor(DEPARTMENT_COLORS[self.dept_index % len(DEPARTMENT_COLORS)])
        self.setBrush(QBrush(color))
        name = DEPARTMENT_NAMES[self.dept_index % len(DEPARTMENT_NAMES)]
        self.setToolTip(name)
