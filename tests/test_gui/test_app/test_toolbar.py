from biosim.gui.controls.toolbar import ControlToolbar


def test_toolbar_creation(qtbot):
    toolbar = ControlToolbar()
    qtbot.addWidget(toolbar)
    assert toolbar.play_btn is not None
    assert toolbar.step_btn is not None
    assert toolbar.speed_slider.value() == 1


def test_toolbar_speed_slider(qtbot):
    toolbar = ControlToolbar()
    qtbot.addWidget(toolbar)

    speeds = []
    toolbar.speed_changed.connect(lambda s: speeds.append(s))

    toolbar.speed_slider.setValue(10)
    assert toolbar.speed_value_label.text() == "10x"
    assert speeds[-1] == 10


def test_toolbar_play_pause_toggle(qtbot):
    toolbar = ControlToolbar()
    qtbot.addWidget(toolbar)

    plays = []
    pauses = []
    toolbar.play_clicked.connect(lambda: plays.append(True))
    toolbar.pause_clicked.connect(lambda: pauses.append(True))

    toolbar.play_btn.click()
    assert len(plays) == 1
    assert toolbar.play_btn.text() == "Pause"

    toolbar.play_btn.click()
    assert len(pauses) == 1
    assert toolbar.play_btn.text() == "Play"
