from biosim.gui.worker import SimulationWorker


def test_worker_creation():
    worker = SimulationWorker()
    assert worker.model is not None
    assert not worker.isRunning()


def test_worker_speed_bounds():
    worker = SimulationWorker()
    worker.set_speed(0)
    assert worker._speed == 1
    worker.set_speed(100)
    assert worker._speed == 20


def test_worker_step_once(qtbot):
    worker = SimulationWorker()
    worker.model.add_company("Test", "#FF0000")

    results = []
    worker.state_updated.connect(lambda s: results.append(s))

    with qtbot.waitSignal(worker.state_updated, timeout=3000):
        worker.start()

    worker.stop()
    worker.wait(2000)

    assert len(results) > 0
    assert "company_names" in results[0]
