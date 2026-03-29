"""Фоновые задачи и виджет холста для отображения графиков.

UI не должен блокироваться во время расчётов и рендеринга фигур, поэтому
здесь лежит минимальная обвязка над `QThread`. Отдельно вынесен `PlotCanvas`,
который аккуратно работает внутри прокручиваемой области интерфейса.
"""

from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QSizePolicy, QWidget


class PlotCanvas(FigureCanvasQTAgg):
    """Matplotlib-холст, адаптированный под прокручиваемую область Qt."""
    def __init__(self, figure: Figure, parent: Optional[QWidget] = None):
        """Встраивает фигуру в Qt-виджет и настраивает политику размеров."""
        super().__init__(figure)
        self.setParent(parent)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.updateGeometry()

    def wheelEvent(self, event) -> None:
        """Прокидывает колесо мыши в родительский `QScrollArea`, если он есть.

        Без этого при наведении курсора на график начинает зумиться сам холст,
        а не прокручиваться страница с результатами, что неудобно для длинных
        отчётов.
        """
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()

        if isinstance(parent, QScrollArea):
            delta_y = event.angleDelta().y()
            if delta_y:
                scrollbar = parent.verticalScrollBar()
                step = max(scrollbar.singleStep(), 24)
                scrollbar.setValue(scrollbar.value() - int((delta_y / 120) * step * 3))
                event.accept()
                return

        super().wheelEvent(event)
