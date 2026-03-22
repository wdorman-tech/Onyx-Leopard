from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QToolBar, QWidget

BUTTON_STYLE = """
    QPushButton {
        background: #334155; color: #e0e0e0; border: 1px solid #4a5568;
        border-radius: 4px; padding: 6px 16px; font-size: 13px;
    }
    QPushButton:hover { background: #4a5568; }
    QPushButton:pressed { background: #2d3748; }
"""

SLIDER_STYLE = """
    QSlider::groove:horizontal { background: #334155; height: 6px; border-radius: 3px; }
    QSlider::handle:horizontal {
        background: #3498DB; width: 14px; margin: -4px 0; border-radius: 7px;
    }
"""


class ControlToolbar(QToolBar):
    """Simulation playback controls: Play/Pause, Step, Speed slider, Tick counter."""

    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    step_clicked = pyqtSignal()
    speed_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        self.setMovable(False)
        self._is_playing = False

        self.play_btn = QPushButton("Play")
        self.play_btn.setStyleSheet(BUTTON_STYLE)
        self.play_btn.clicked.connect(self._toggle_play)
        self.addWidget(self.play_btn)

        self.step_btn = QPushButton("Step")
        self.step_btn.setStyleSheet(BUTTON_STYLE)
        self.step_btn.clicked.connect(self.step_clicked.emit)
        self.addWidget(self.step_btn)

        self.addSeparator()

        speed_widget = QWidget()
        speed_layout = QHBoxLayout(speed_widget)
        speed_layout.setContentsMargins(4, 0, 4, 0)

        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: #e0e0e0;")
        speed_layout.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(1)
        self.speed_slider.setFixedWidth(120)
        self.speed_slider.setStyleSheet(SLIDER_STYLE)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        speed_layout.addWidget(self.speed_slider)

        self.speed_value_label = QLabel("1x")
        self.speed_value_label.setStyleSheet("color: #e0e0e0; min-width: 30px;")
        speed_layout.addWidget(self.speed_value_label)

        self.addWidget(speed_widget)
        self.addSeparator()

        self.tick_label = QLabel("Tick: 0")
        self.tick_label.setStyleSheet(
            "color: #3498DB; font-size: 14px; font-weight: bold; padding: 0 12px;"
        )
        self.addWidget(self.tick_label)

    def _toggle_play(self):
        if self._is_playing:
            self._is_playing = False
            self.play_btn.setText("Play")
            self.pause_clicked.emit()
        else:
            self._is_playing = True
            self.play_btn.setText("Pause")
            self.play_clicked.emit()

    def _on_speed_change(self, value: int):
        self.speed_value_label.setText(f"{value}x")
        self.speed_changed.emit(value)

    def update_tick_display(self, tick: int):
        self.tick_label.setText(f"Tick: {tick}")
