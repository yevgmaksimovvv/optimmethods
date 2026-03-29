"""Переиспользуемые helper-функции для Qt layout."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QSplitter, QWidget


def create_scroll_container(
    content: QWidget,
    *,
    widget_resizable: bool = True,
    horizontal_policy: Qt.ScrollBarPolicy = Qt.ScrollBarAlwaysOff,
) -> QScrollArea:
    """Оборачивает widget в QScrollArea со стандартной конфигурацией."""
    container = QScrollArea()
    container.setWidgetResizable(widget_resizable)
    container.setHorizontalScrollBarPolicy(horizontal_policy)
    container.setFrameShape(QScrollArea.NoFrame)
    container.setWidget(content)
    return container


def configure_two_panel_splitter(
    splitter: QSplitter,
    *,
    left: QWidget,
    right: QWidget,
    left_size: int,
    right_size: int,
    handle_width: int | None = None,
) -> None:
    """Стандартизирует настройку двухпанельного splitter."""
    splitter.setChildrenCollapsible(False)
    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([left_size, right_size])
    if handle_width is not None:
        splitter.setHandleWidth(handle_width)
