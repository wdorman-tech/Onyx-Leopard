from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DashboardPanel(QWidget):
    """Stub -- real implementation in Unit 5."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("Dashboard (loading...)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #e0e0e0; font-size: 16px;")
        layout.addWidget(label)

    def update_state(self, state: dict):
        pass

    def clear_data(self):
        pass
