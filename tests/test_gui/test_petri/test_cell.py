from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt

from biosim.gui.petri.cell import DEPARTMENT_COLORS, CellItem


def test_cell_correct_color_for_each_department(qtbot):
    for dept_idx in range(12):
        cell = CellItem(dept_idx, organism_radius=50.0)
        brush_color = cell.brush().color().name()
        assert brush_color == DEPARTMENT_COLORS[dept_idx].lower()


def test_cell_wraps_department_index(qtbot):
    # Index beyond 12 should wrap
    cell = CellItem(13, organism_radius=50.0)
    brush_color = cell.brush().color().name()
    assert brush_color == DEPARTMENT_COLORS[1].lower()


def test_cell_seed_spawn_near_center(qtbot):
    """With seed_spawn=True (default), cells spawn within 15% of radius."""
    radius = 60.0
    for _ in range(50):
        cell = CellItem(0, organism_radius=radius, seed_spawn=True)
        dist = np.sqrt(cell.x() ** 2 + cell.y() ** 2)
        assert dist <= radius * 0.15 + 1.0


def test_cell_legacy_spawn_wider(qtbot):
    """With seed_spawn=False, cells spawn within 70% of radius."""
    radius = 60.0
    for _ in range(50):
        cell = CellItem(0, organism_radius=radius, seed_spawn=False)
        dist = np.sqrt(cell.x() ** 2 + cell.y() ** 2)
        assert dist <= radius * 0.7 + 1.0


def test_cell_initial_position_within_organism(qtbot):
    """Default seed_spawn=True keeps cells near center."""
    radius = 60.0
    for _ in range(50):
        cell = CellItem(0, organism_radius=radius)
        dist = np.sqrt(cell.x() ** 2 + cell.y() ** 2)
        assert dist <= radius * 0.15 + 1.0


def test_cell_has_no_pen(qtbot):
    cell = CellItem(0, organism_radius=30.0)
    assert cell.pen().style() == Qt.PenStyle.NoPen


def test_cell_no_brownian_step_method(qtbot):
    """brownian_step was removed — physics are now handled by ClusterPhysics."""
    cell = CellItem(0, organism_radius=30.0)
    assert not hasattr(cell, "brownian_step")
