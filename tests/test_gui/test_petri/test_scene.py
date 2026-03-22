from biosim.gui.petri.scene import PetriDishScene


def _make_state(n: int = 3) -> dict:
    """Build a minimal state dict for n companies."""
    return {
        "n_active": n,
        "indices": list(range(n)),
        "company_names": [f"Company_{i}" for i in range(n)],
        "company_colors": ["#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#E67E22"][:n],
        "firm_size": [100.0] * n,
        "health_score": [0.8] * n,
        "revenue": [1_000_000.0] * n,
        "dept_headcount": [[10.0] * 12 for _ in range(n)],
    }


def test_scene_creation(qtbot):
    scene = PetriDishScene()
    assert scene.sceneRect().width() == 1000
    assert scene.sceneRect().height() == 1000
    # Dish border should be drawn (at least 1 item)
    assert len(scene.items()) > 0


def test_update_state_adds_organisms(qtbot):
    scene = PetriDishScene()
    state = _make_state(3)
    scene.update_state(state)
    assert len(scene.organisms) == 3


def test_update_state_five_organisms(qtbot):
    scene = PetriDishScene()
    state = _make_state(5)
    scene.update_state(state)
    assert len(scene.organisms) == 5


def test_update_state_removes_dead_organism(qtbot):
    scene = PetriDishScene()
    state = _make_state(3)
    scene.update_state(state)
    assert len(scene.organisms) == 3

    # Remove company index 2
    reduced = _make_state(2)
    scene.update_state(reduced)
    assert len(scene.organisms) == 2
    assert 2 not in scene.organisms


def test_organisms_have_correct_cell_count(qtbot):
    scene = PetriDishScene()
    state = _make_state(1)
    # 12 departments, each with 10 headcount -> should produce cells
    state["dept_headcount"] = [[10.0] * 12]
    scene.update_state(state)

    organism = scene.organisms[0]
    assert len(organism.cells) > 0
    # With 120 total headcount and target ~50 cells, each dept gets ~4 cells
    # but at least 1 per dept with headcount > 0
    assert len(organism.cells) >= 12


def test_update_state_with_empty(qtbot):
    scene = PetriDishScene()
    state = {
        "n_active": 0,
        "indices": [],
        "company_names": [],
        "company_colors": [],
        "firm_size": [],
        "health_score": [],
        "revenue": [],
        "dept_headcount": [],
    }
    scene.update_state(state)
    assert len(scene.organisms) == 0
