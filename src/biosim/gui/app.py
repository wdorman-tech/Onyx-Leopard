import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar, QTabWidget

from biosim.gui.controls.toolbar import ControlToolbar
from biosim.gui.dashboard.panel import DashboardPanel
from biosim.gui.petri.scene import PetriDishView
from biosim.gui.worker import SimulationWorker

DARK_THEME = """
    QMainWindow { background-color: #0f0f23; }
    QTabWidget::pane { border: 1px solid #334155; background: #0f0f23; }
    QTabBar::tab {
        background: #1a1a2e; color: #e0e0e0; padding: 8px 20px;
        border: 1px solid #334155; border-bottom: none;
    }
    QTabBar::tab:selected { background: #0f0f23; border-bottom: 2px solid #3498DB; }
    QStatusBar { background: #1a1a2e; color: #e0e0e0; }
    QMenuBar { background: #1a1a2e; color: #e0e0e0; }
    QMenuBar::item:selected { background: #334155; }
    QToolBar { background: #1a1a2e; border: none; spacing: 8px; }
"""

COMPANY_PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

DEFAULT_COMPANIES = [
    ("Alpha Corp", COMPANY_PALETTE[0], "technology", "medium"),
    ("Beta Inc", COMPANY_PALETTE[1], "manufacturing", "large"),
    ("Gamma Ltd", COMPANY_PALETTE[2], "services", "small"),
]


class MainWindow(QMainWindow):
    """BioSim main window with petri dish and dashboard tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioSim — Biological Business Simulator")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self.setStyleSheet(DARK_THEME)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.petri_view = PetriDishView()
        self.dashboard = DashboardPanel()
        self.tabs.addTab(self.petri_view, "Petri Dish")
        self.tabs.addTab(self.dashboard, "Dashboard")

        self._setup_menus()

        self.control_toolbar = ControlToolbar()
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.control_toolbar)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready -- Press Play to start simulation")

        self.worker = SimulationWorker()
        self._connect_worker(self.worker)
        self._connect_toolbar()
        self._init_default_simulation()

    def _setup_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        new_action = QAction("New Simulation", self)
        new_action.triggered.connect(self._new_simulation)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _connect_worker(self, worker: SimulationWorker):
        worker.state_updated.connect(self._on_state_update)
        worker.tick_updated.connect(self._on_tick_update)

    def _connect_toolbar(self):
        """One-time toolbar signal wiring -- delegates through self to current worker."""
        self.control_toolbar.play_clicked.connect(self._on_play)
        self.control_toolbar.pause_clicked.connect(self._on_pause)
        self.control_toolbar.step_clicked.connect(self._on_step)
        self.control_toolbar.speed_changed.connect(self._on_speed_change)

    def _init_default_simulation(self):
        for name, color, industry, size in DEFAULT_COMPANIES:
            self.worker.model.add_company(name, color, industry=industry, size=size)

    def _new_simulation(self):
        self.worker.stop()
        self.worker.wait(2000)
        self.worker = SimulationWorker()
        self._connect_worker(self.worker)
        self.dashboard.clear_data()
        self._init_default_simulation()
        self.status_bar.showMessage("New simulation created")

    def _on_play(self):
        if not self.worker.isRunning():
            self.worker.start()
        else:
            self.worker.resume()

    def _on_pause(self):
        self.worker.pause()

    def _on_step(self):
        self.worker.step_once()

    def _on_speed_change(self, speed: int):
        self.worker.set_speed(speed)

    def _on_state_update(self, state: dict):
        self.petri_view.scene.update_state(state)
        self.dashboard.update_state(state)

    def _on_tick_update(self, tick: int):
        self.status_bar.showMessage(f"Tick {tick} | Simulation running")
        self.control_toolbar.update_tick_display(tick)

    def closeEvent(self, event):  # noqa: N802
        self.worker.stop()
        self.worker.wait(2000)
        super().closeEvent(event)


def main():
    """Entry point for `python -m biosim`."""
    app = QApplication(sys.argv)
    app.setApplicationName("BioSim")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
