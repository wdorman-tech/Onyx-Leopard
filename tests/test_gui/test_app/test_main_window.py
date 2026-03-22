from biosim.gui.app import MainWindow


def test_main_window_creation(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "BioSim \u2014 Biological Business Simulator"
    assert window.tabs.count() == 2


def test_main_window_has_controls(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.control_toolbar is not None
    assert window.status_bar is not None


def test_default_companies_created(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert len(window.worker.model.agents_list) == 3
