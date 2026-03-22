from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItemGroup, QGraphicsPathItem, QGraphicsSimpleTextItem
from scipy.spatial import ConvexHull

from biosim.gui.petri.cell import CellItem


class OrganismBoundary(QGraphicsPathItem):
    """Smoothed convex hull boundary around an organism's cells."""

    def __init__(self, color: str) -> None:
        super().__init__()
        self._set_style(color, 1.0)

    def _set_style(self, color: str, health: float) -> None:
        pen_color = QColor(color)
        pen_color.setAlphaF(0.6 * health)
        self.setPen(QPen(pen_color, 2.0))
        fill_color = QColor(color)
        fill_color.setAlphaF(0.08)
        self.setBrush(QBrush(fill_color))

    def update_from_cells(
        self,
        cells: list[CellItem],
        radius: float,
        health_score: float,
        color: str,
    ) -> None:
        """Redraw boundary as smoothed convex hull of cell positions."""
        self._set_style(color, max(0.2, health_score))

        if len(cells) < 3:
            path = QPainterPath()
            path.addEllipse(-radius, -radius, radius * 2, radius * 2)
            self.setPath(path)
            return

        points = np.array([[c.x(), c.y()] for c in cells])

        try:
            hull = ConvexHull(points)
        except Exception:
            path = QPainterPath()
            path.addEllipse(-radius, -radius, radius * 2, radius * 2)
            self.setPath(path)
            return

        hull_pts = points[hull.vertices]
        path = _smooth_hull_path(hull_pts, padding=6.0)
        self.setPath(path)


class OrganismItem(QGraphicsItemGroup):
    """A company organism rendered as a blob containing colored cells."""

    def __init__(
        self,
        company_index: int,
        name: str,
        color: str,
        position: tuple[float, float],
    ) -> None:
        super().__init__()
        self.company_index = company_index
        self.name = name
        self.color = color
        self.boundary = OrganismBoundary(color)
        self.addToGroup(self.boundary)
        self.cells: list[CellItem] = []
        self._label = QGraphicsSimpleTextItem(name)
        self._label.setBrush(QBrush(QColor("#cccccc")))
        self._label.setPos(-len(name) * 3, 0)
        self.addToGroup(self._label)
        self.setPos(position[0], position[1])
        self.setToolTip(name)
        self._current_radius = 30.0

    def update_from_state(
        self,
        firm_size: float,
        health_score: float,
        revenue: float,
        dept_headcount: list[float],
    ) -> None:
        """Update organism visuals from simulation state data."""
        radius = 20 + 30 * np.log1p(firm_size)
        self._current_radius = radius

        self._sync_cells(dept_headcount, radius)
        self._update_boundary(radius, health_score)

        total_headcount = sum(dept_headcount)
        self.setToolTip(
            f"{self.name}\n"
            f"Revenue: ${revenue:,.0f}\n"
            f"Headcount: {total_headcount:.0f}\n"
            f"Health: {health_score:.0%}"
        )
        self._label.setPos(-len(self.name) * 3, radius + 8)

    def _sync_cells(self, dept_headcount: list[float], radius: float) -> None:
        """Add/remove cells to match current department headcounts."""
        total = sum(dept_headcount)
        target_cells: list[int] = []
        for dept_idx, count in enumerate(dept_headcount):
            if total <= 0:
                continue
            n_cells = max(0, int(round(count / max(1, total / 50))))
            if count > 0 and n_cells == 0:
                n_cells = 1
            target_cells.extend([dept_idx] * n_cells)

        while len(self.cells) > len(target_cells):
            cell = self.cells.pop()
            self.removeFromGroup(cell)
            cell.setParentItem(None)
            if cell.scene():
                cell.scene().removeItem(cell)

        while len(self.cells) < len(target_cells):
            dept_idx = target_cells[len(self.cells)]
            cell = CellItem(dept_idx, radius)
            self.cells.append(cell)
            self.addToGroup(cell)

        for i, cell in enumerate(self.cells):
            if i < len(target_cells):
                cell.dept_index = target_cells[i]
                cell.update_color()
                cell.brownian_step(radius)

    def _update_boundary(self, radius: float, health_score: float) -> None:
        self.boundary.update_from_cells(self.cells, radius, health_score, self.color)


def _smooth_hull_path(hull_pts: np.ndarray, padding: float = 6.0) -> QPainterPath:
    """Create a smooth Bezier path from convex hull points with outward padding."""
    n = len(hull_pts)
    if n < 3:
        path = QPainterPath()
        path.addEllipse(-20, -20, 40, 40)
        return path

    centroid = hull_pts.mean(axis=0)
    padded = []
    for pt in hull_pts:
        direction = pt - centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        padded.append(pt + direction * padding)
    padded = np.array(padded)

    path = QPainterPath()
    mid0 = (padded[0] + padded[1]) / 2
    path.moveTo(QPointF(float(mid0[0]), float(mid0[1])))

    for i in range(n):
        p1 = padded[(i + 1) % n]
        p2 = padded[(i + 2) % n]
        mid = (p1 + p2) / 2
        path.quadTo(
            QPointF(float(p1[0]), float(p1[1])),
            QPointF(float(mid[0]), float(mid[1])),
        )

    path.closeSubpath()
    return path
