from PyQt6.QtCore import QMutex, QThread, QWaitCondition, pyqtSignal

from biosim.agents.mesa_model import BioSimModel


class SimulationWorker(QThread):
    """Runs the simulation tick loop on a background thread."""

    state_updated = pyqtSignal(dict)
    tick_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.model = BioSimModel()
        self._running = False
        self._paused = False
        self._step_once_flag = False
        self._speed = 1
        self._mutex = QMutex()
        self._condition = QWaitCondition()

    def run(self):
        self._running = True
        self._paused = False

        while self._running:
            self._mutex.lock()
            if self._paused and not self._step_once_flag:
                self._condition.wait(self._mutex)
            step_once = self._step_once_flag
            self._step_once_flag = False
            self._mutex.unlock()

            if not self._running:
                break

            self.model.step()
            snapshot = self.model.last_snapshot.copy()
            tick = self.model.state_manager.tick_count
            snapshot["tick"] = tick

            self.state_updated.emit(snapshot)
            self.tick_updated.emit(tick)

            if step_once:
                self._paused = True
                continue

            interval_ms = max(50, 1000 // self._speed)
            self.msleep(interval_ms)

    def pause(self):
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self._paused = False
        self._condition.wakeAll()
        self._mutex.unlock()

    def step_once(self):
        self._mutex.lock()
        self._step_once_flag = True
        self._paused = False
        self._condition.wakeAll()
        self._mutex.unlock()

    def set_speed(self, speed: int):
        self._speed = max(1, min(20, speed))

    def stop(self):
        self._running = False
        self._mutex.lock()
        self._paused = False
        self._condition.wakeAll()
        self._mutex.unlock()
        self.quit()
