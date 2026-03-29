"""Общие Qt-виджеты заголовков таблиц с HTML-форматированием."""

from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QPainter, QTextDocument
from PySide6.QtWidgets import QHeaderView, QStyle, QStyleOptionHeader, QWidget


class MathHeaderView(QHeaderView):
    """Header, который рендерит подписи через HTML для математики."""

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None = None):
        super().__init__(orientation, parent)
        self._labels: list[str] = []

    def set_math_labels(self, labels: list[str]) -> None:
        self._labels = labels
        self.viewport().update()

    def paintSection(self, painter: QPainter, rect: QRect, logical_index: int) -> None:
        if logical_index < 0:
            return

        option = QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.section = logical_index
        option.text = ""
        option.position = QStyleOptionHeader.SectionPosition.Middle
        option.textAlignment = Qt.AlignCenter
        self.style().drawControl(QStyle.ControlElement.CE_Header, option, painter, self)

        if logical_index >= len(self._labels):
            return
        text = self._labels[logical_index]
        if not text:
            return

        text_rect = rect.adjusted(6, 2, -6, -2)
        doc = QTextDocument(self)
        doc.setHtml(
            "<div style='text-align:center; color:#dde8fa; font-weight:700; font-size:12px;'>"
            f"{text}</div>"
        )
        doc.setTextWidth(text_rect.width())
        content_height = doc.size().height()
        top_shift = max((text_rect.height() - content_height) / 2.0, 0.0)
        painter.save()
        painter.translate(text_rect.left(), text_rect.top() + top_shift)
        doc.drawContents(painter, QRectF(0, 0, text_rect.width(), text_rect.height()))
        painter.restore()
