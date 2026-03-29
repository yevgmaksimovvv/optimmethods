"""Общий конструктор левой панели и секций управления."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class ControlsPanel:
    """Собранная панель управления и её корневой layout."""

    panel: QWidget
    layout: QVBoxLayout


def create_controls_panel(
    *,
    min_width: int = 500,
    max_width: int = 560,
    spacing: int = 12,
) -> ControlsPanel:
    """Создаёт стандартную левую панель управления."""
    panel = QWidget()
    panel.setMinimumWidth(min_width)
    panel.setMaximumWidth(max_width)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    return ControlsPanel(panel=panel, layout=layout)


def create_standard_group(
    title: str,
    *,
    spacing: int = 8,
    margins: tuple[int, int, int, int] = (16, 16, 16, 14),
) -> tuple[QGroupBox, QVBoxLayout]:
    """Создаёт стандартную секцию `QGroupBox` с типовыми отступами."""
    group = QGroupBox(title)
    group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return group, layout


def create_primary_action_button(
    *,
    text: str,
    on_click: Callable[[], None],
    min_height: int = 42,
    role: str = "action",
) -> QPushButton:
    """Создаёт primary-кнопку запуска действия."""
    button = QPushButton(text)
    button.clicked.connect(on_click)
    button.setProperty("variant", "primary")
    button.setProperty("role", role)
    button.setMinimumHeight(min_height)
    return button


def create_flush_row(*widgets: QWidget, spacing: int = 8) -> tuple[QWidget, QHBoxLayout]:
    """Создаёт строку с нулевыми внешними отступами."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return row, layout


def create_choice_chip_grid(
    *,
    group: QButtonGroup,
    options: Sequence[tuple[str, str]],
    columns: int,
    role: str = "choice-chip",
    horizontal_spacing: int = 8,
    vertical_spacing: int = 8,
    on_clicked: Callable[[str, bool], None] | None = None,
    tooltips: dict[str, str] | None = None,
) -> tuple[QWidget, list[QPushButton]]:
    """Создаёт сетку checkable chip-кнопок и регистрирует их в `QButtonGroup`."""
    holder = QWidget()
    layout = QGridLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(horizontal_spacing)
    layout.setVerticalSpacing(vertical_spacing)
    layout.setAlignment(Qt.AlignTop)
    for column in range(max(columns, 1)):
        layout.setColumnStretch(column, 1)

    buttons: list[QPushButton] = []
    for index, (caption, value) in enumerate(options):
        button = QPushButton(caption)
        button.setCheckable(True)
        button.setProperty("role", role)
        button.setProperty("choice_value", value)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if tooltips is not None and value in tooltips:
            button.setToolTip(tooltips[value])
        if on_clicked is not None:
            button.clicked.connect(partial(on_clicked, value))
        group.addButton(button)
        layout.addWidget(button, index // columns, index % columns)
        buttons.append(button)
    return holder, buttons


def create_parameter_grid(
    group: QGroupBox,
    *,
    label_min_width: int = 150,
    horizontal_spacing: int = 10,
    vertical_spacing: int = 8,
    margins: tuple[int, int, int, int] = (16, 16, 16, 14),
) -> QGridLayout:
    """Создаёт унифицированную сетку для секции параметров."""
    grid = QGridLayout(group)
    grid.setContentsMargins(*margins)
    grid.setHorizontalSpacing(horizontal_spacing)
    grid.setVerticalSpacing(vertical_spacing)
    grid.setColumnMinimumWidth(0, label_min_width)
    grid.setColumnStretch(1, 0)
    grid.setColumnStretch(2, 1)
    return grid


def add_parameter_row(
    grid: QGridLayout,
    *,
    row: int,
    label: str,
    control: QWidget,
    label_role: str = "parameter-label",
    control_colspan: int = 2,
    label_position: str = "left",
) -> QLabel:
    """Добавляет строку параметра вида `label | control`."""
    if label_position not in {"left", "top"}:
        raise ValueError("label_position должен быть 'left' или 'top'")
    label_widget = QLabel(label)
    label_widget.setProperty("role", label_role)
    label_widget.setMinimumWidth(150)
    label_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    label_widget.setWordWrap(True)
    if label_position == "left":
        grid.addWidget(label_widget, row, 0)
        grid.addWidget(control, row, 1, 1, control_colspan)
    else:
        base_row = row * 2
        grid.addWidget(label_widget, base_row, 0, 1, 3)
        grid.addWidget(control, base_row + 1, 0, 1, 3)
    return label_widget
