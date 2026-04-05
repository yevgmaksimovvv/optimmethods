"""Переиспользуемые helper-функции для QTableWidget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget


def configure_data_table(
    table: QTableWidget,
    *,
    min_row_height: int,
    allow_selection: bool,
    allow_editing: bool,
    word_wrap: bool,
) -> None:
    """Применяет базовую конфигурацию таблицы с данными.

    Контракт:
    - `allow_selection` управляет только режимом выделения строк;
    - `allow_editing` управляет только редактированием ячеек.
    """
    table.setAlternatingRowColors(True)
    table.setWordWrap(word_wrap)
    table.setTextElideMode(Qt.ElideRight)
    table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
    table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
    table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
    table.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
    table.verticalHeader().setDefaultSectionSize(min_row_height)
    table.verticalHeader().setMinimumSectionSize(min_row_height)

    if allow_selection:
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
    else:
        table.setSelectionMode(QAbstractItemView.NoSelection)

    if allow_editing:
        table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
        )
    else:
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

    if allow_selection or allow_editing:
        table.setFocusPolicy(Qt.StrongFocus)
    else:
        table.setFocusPolicy(Qt.NoFocus)


def set_table_empty_layout(table: QTableWidget) -> None:
    """Пустая таблица должна равномерно занимать ширину."""
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


def set_table_data_layout(table: QTableWidget, min_widths: list[int]) -> None:
    """Таблица с данными: ширина по содержимому + минимумы колонок."""
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Interactive)
    table.resizeColumnsToContents()
    for column, min_width in enumerate(min_widths):
        table.setColumnWidth(column, max(table.columnWidth(column), min_width))
