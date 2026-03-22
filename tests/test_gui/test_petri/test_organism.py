from biosim.gui.petri.organism import OrganismItem


def test_organism_creation(qtbot):
    org = OrganismItem(0, "TestCorp", "#E74C3C", (100.0, 50.0))
    assert org.company_index == 0
    assert org.name == "TestCorp"
    assert org.color == "#E74C3C"
    assert org.x() == 100.0
    assert org.y() == 50.0


def test_organism_position(qtbot):
    org = OrganismItem(1, "Acme", "#3498DB", (-200.0, 150.0))
    assert org.x() == -200.0
    assert org.y() == 150.0


def test_update_from_state_changes_cell_count(qtbot):
    org = OrganismItem(0, "TestCorp", "#E74C3C", (0.0, 0.0))
    assert len(org.cells) == 0

    # Give it some headcount
    dept_hc = [10.0] * 12
    org.update_from_state(firm_size=50.0, health_score=0.9, revenue=500000, dept_headcount=dept_hc)
    initial_count = len(org.cells)
    assert initial_count > 0

    # Increase headcount substantially
    dept_hc_big = [100.0] * 12
    org.update_from_state(
        firm_size=200.0, health_score=0.9, revenue=2000000, dept_headcount=dept_hc_big
    )
    # Should still be around 50 cells (capped) but at least as many departments
    assert len(org.cells) >= 12


def test_update_from_state_shrinks_cells(qtbot):
    org = OrganismItem(0, "TestCorp", "#E74C3C", (0.0, 0.0))
    dept_hc = [20.0] * 12
    org.update_from_state(
        firm_size=100.0, health_score=0.8, revenue=1000000, dept_headcount=dept_hc
    )
    big_count = len(org.cells)

    # Shrink: only 2 departments have headcount
    dept_hc_small = [0.0] * 12
    dept_hc_small[0] = 5.0
    dept_hc_small[1] = 5.0
    org.update_from_state(
        firm_size=10.0, health_score=0.5, revenue=100000, dept_headcount=dept_hc_small
    )
    assert len(org.cells) < big_count


def test_tooltip_shows_correct_info(qtbot):
    org = OrganismItem(0, "TestCorp", "#E74C3C", (0.0, 0.0))
    dept_hc = [10.0] * 12
    org.update_from_state(
        firm_size=50.0, health_score=0.75, revenue=1234567, dept_headcount=dept_hc
    )
    tip = org.toolTip()
    assert "TestCorp" in tip
    assert "1,234,567" in tip
    assert "120" in tip  # total headcount
    assert "75" in tip  # health percentage


def test_zero_headcount_no_cells(qtbot):
    org = OrganismItem(0, "Ghost", "#888888", (0.0, 0.0))
    dept_hc = [0.0] * 12
    org.update_from_state(firm_size=1.0, health_score=0.1, revenue=0, dept_headcount=dept_hc)
    assert len(org.cells) == 0
