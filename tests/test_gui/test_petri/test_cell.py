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


def test_brownian_step_keeps_within_radius(qtbot):
    radius = 40.0
    cell = CellItem(0, organism_radius=radius)
    # Run many brownian steps and verify cell stays within bounds
    for _ in range(200):
        cell.brownian_step(radius)
        dist = np.sqrt(cell.x() ** 2 + cell.y() ** 2)
        assert dist <= radius * 0.75 + 5.0  # small tolerance for float


def test_cell_initial_position_within_organism(qtbot):
    radius = 60.0
    for _ in range(50):
        cell = CellItem(0, organism_radius=radius)
        dist = np.sqrt(cell.x() ** 2 + cell.y() ** 2)
        assert dist <= radius * 0.7 + 1.0


def test_cell_has_no_pen(qtbot):
    cell = CellItem(0, organism_radius=30.0)
    assert cell.pen().style() == Qt.PenStyle.NoPen
