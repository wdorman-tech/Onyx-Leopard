from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView


class PetriDishView(QGraphicsView):
    """Stub -- real implementation in Unit 4."""

    def __init__(self, parent=None):
        self.scene = PetriDishScene()
        super().__init__(self.scene, parent)
        self.setBackgroundBrush(QBrush(QColor("#1a1a2e")))


class PetriDishScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setSceneRect(-500, -500, 1000, 1000)

    def update_state(self, state: dict):
        pass
