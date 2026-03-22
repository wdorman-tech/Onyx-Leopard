from __future__ import annotations

from PyQt6.QtCore import QTimeLine
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsScene

from biosim.gui.petri.cell import CellItem


class CellDivisionEffect:
    """Animate a cell elongating, pinching, and splitting into two."""

    DURATION_MS = 600

    def __init__(self, cell: CellItem, scene: QGraphicsScene) -> None:
        self._cell = cell
        self._scene = scene
        self._timeline = QTimeLine(self.DURATION_MS)
        self._timeline.setFrameRange(0, 100)
        self._original_rect = cell.rect()
        self._timeline.frameChanged.connect(self._step)
        self._timeline.finished.connect(self._finish)
        self._split_cell: CellItem | None = None

    def start(self) -> None:
        self._timeline.start()

    def _step(self, frame: int) -> None:
        progress = frame / 100.0
        r = self._original_rect
        if progress < 0.5:
            stretch = 1.0 + progress * 1.5
            self._cell.setRect(
                r.x() * stretch,
                r.y(),
                r.width() * stretch,
                r.height(),
            )
        else:
            pinch = (progress - 0.5) * 2
            opacity = 1.0 - pinch * 0.3
            self._cell.setOpacity(max(0.5, opacity))

    def _finish(self) -> None:
        self._cell.setRect(self._original_rect)
        self._cell.setOpacity(1.0)


class CellApoptosisEffect:
    """Animate a cell shrinking and fading out before removal."""

    DURATION_MS = 800

    def __init__(self, cell: CellItem, scene: QGraphicsScene) -> None:
        self._cell = cell
        self._scene = scene
        self._timeline = QTimeLine(self.DURATION_MS)
        self._timeline.setFrameRange(0, 100)
        self._original_rect = cell.rect()
        self._original_color = cell.brush().color()
        self._timeline.frameChanged.connect(self._step)
        self._timeline.finished.connect(self._finish)

    def start(self) -> None:
        self._timeline.start()

    def _step(self, frame: int) -> None:
        progress = frame / 100.0
        scale = 1.0 - progress * 0.8
        r = self._original_rect
        self._cell.setRect(
            r.x() * scale,
            r.y() * scale,
            r.width() * scale,
            r.height() * scale,
        )
        gray = QColor.fromHslF(0, 0, 0.5, 1.0 - progress)
        blended = _blend_colors(self._original_color, gray, progress)
        self._cell.setBrush(QBrush(blended))
        self._cell.setOpacity(1.0 - progress)

    def _finish(self) -> None:
        if self._cell.scene():
            self._scene.removeItem(self._cell)


class BoundaryFlashEffect:
    """Briefly flash an organism boundary red (competition overlap)."""

    DURATION_MS = 400

    def __init__(self, boundary_item: object) -> None:
        self._item = boundary_item
        self._original_pen: QPen | None = None
        self._timeline = QTimeLine(self.DURATION_MS)
        self._timeline.setFrameRange(0, 100)
        self._timeline.frameChanged.connect(self._step)
        self._timeline.finished.connect(self._finish)

    def start(self) -> None:
        if hasattr(self._item, "pen"):
            self._original_pen = self._item.pen()
        self._timeline.start()

    def _step(self, frame: int) -> None:
        progress = frame / 100.0
        if not hasattr(self._item, "setPen"):
            return
        alpha = 1.0 - progress
        flash_color = QColor(231, 76, 60, int(alpha * 255))
        self._item.setPen(QPen(flash_color, 3.0))

    def _finish(self) -> None:
        if self._original_pen and hasattr(self._item, "setPen"):
            self._item.setPen(self._original_pen)


def _blend_colors(c1: QColor, c2: QColor, t: float) -> QColor:
    """Linearly interpolate between two QColors."""
    t = max(0.0, min(1.0, t))
    return QColor(
        int(c1.red() + (c2.red() - c1.red()) * t),
        int(c1.green() + (c2.green() - c1.green()) * t),
        int(c1.blue() + (c2.blue() - c1.blue()) * t),
        int(c1.alpha() + (c2.alpha() - c1.alpha()) * t),
    )
