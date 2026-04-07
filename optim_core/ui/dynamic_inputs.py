"""Переиспользуемые контролы динамического ввода значений через `+/-`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)


@dataclass(frozen=True)
class DynamicInputEntry:
    """Один элемент динамического ряда."""

    widget: QWidget
    fields: tuple[QLineEdit, ...]
    remove_button: QPushButton


class DynamicSeriesInputRow:
    """Динамический ряд полей с кнопками `+`/`-`."""

    def __init__(
        self,
        *,
        add_role: str,
        remove_role: str,
        field_role: str,
        placeholders: tuple[str, ...],
        field_widths: tuple[int, ...],
        control_button_size: int,
        row_control_spacing: int,
        add_button_text: str = "+",
        remove_button_text: str = "−",
        separator_text: str | None = None,
        separator_role: str | None = None,
        scroll_min_height: int = 72,
        scroll_max_height: int = 76,
        container_min_height: int = 56,
        min_items: int = 1,
        on_control_added: Callable[[QWidget], None] | None = None,
        on_control_removed: Callable[[QWidget], None] | None = None,
    ) -> None:
        if not placeholders:
            raise ValueError("placeholders не может быть пустым")
        if len(placeholders) != len(field_widths):
            raise ValueError("Количество placeholders и field_widths должно совпадать")
        if min_items < 1:
            raise ValueError("min_items должен быть >= 1")

        self._placeholders = placeholders
        self._field_widths = field_widths
        self._control_button_size = control_button_size
        self._row_control_spacing = row_control_spacing
        self._remove_role = remove_role
        self._field_role = field_role
        self._separator_text = separator_text
        self._separator_role = separator_role
        self._scroll_min_height = scroll_min_height
        self._scroll_max_height = scroll_max_height
        self._container_min_height = container_min_height
        self._min_items = min_items
        self._on_control_added = on_control_added
        self._on_control_removed = on_control_removed

        self.entries: list[DynamicInputEntry] = []

        self.row_widget = QWidget()
        row_layout = QHBoxLayout(self.row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.setAlignment(Qt.AlignVCenter)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setMinimumHeight(self._scroll_min_height)
        self.scroll.setMaximumHeight(self._scroll_max_height)
        self.scroll.setMinimumWidth(0)
        self.scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        self.container = QWidget()
        self.fields_layout = QHBoxLayout(self.container)
        self.fields_layout.setContentsMargins(0, 0, 0, 0)
        self.fields_layout.setSpacing(8)
        self.fields_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.fields_layout.setSizeConstraint(QHBoxLayout.SetFixedSize)
        self.scroll.setWidget(self.container)

        self.add_button = QPushButton(add_button_text)
        self.add_button.setProperty("role", add_role)
        self.add_button.setFixedSize(control_button_size, control_button_size)
        self.add_button.clicked.connect(lambda _checked=False: self.add_item())
        if self._on_control_added is not None:
            self._on_control_added(self.add_button)

        row_layout.addWidget(self.scroll, 1, Qt.AlignTop)
        row_layout.addWidget(self.add_button, 0, Qt.AlignTop)
        row_layout.setStretch(0, 1)
        row_layout.setStretch(1, 0)

        self._remove_button_text = remove_button_text

    def add_item(self, *values: str) -> None:
        normalized_values = self._normalize_values(values)
        item = QWidget()
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(self._row_control_spacing)

        fields: list[QLineEdit] = []
        for idx, (placeholder, width) in enumerate(zip(self._placeholders, self._field_widths, strict=True)):
            line = QLineEdit(normalized_values[idx])
            line.setProperty("role", self._field_role)
            line.setPlaceholderText(placeholder)
            line.setFixedWidth(width)
            line.setFixedHeight(self._control_button_size)
            line.setAlignment(Qt.AlignCenter)
            item_layout.addWidget(line)
            fields.append(line)
            if self._separator_text is not None and idx < len(self._placeholders) - 1:
                separator = QLabel(self._separator_text)
                if self._separator_role is not None:
                    separator.setProperty("role", self._separator_role)
                separator.setAlignment(Qt.AlignCenter)
                item_layout.addWidget(separator)

        remove_button = QPushButton(self._remove_button_text)
        remove_button.setProperty("role", self._remove_role)
        remove_button.setFixedSize(self._control_button_size, self._control_button_size)
        first_field = fields[0]
        remove_button.clicked.connect(lambda _checked=False, line=first_field: self.remove_item(line))
        item_layout.addWidget(remove_button)

        item.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        item.setFixedWidth(self._entry_width())
        item.setFixedHeight(self._control_button_size)
        self.fields_layout.addWidget(item)

        entry = DynamicInputEntry(widget=item, fields=tuple(fields), remove_button=remove_button)
        self.entries.append(entry)

        if self._on_control_added is not None:
            for field in fields:
                self._on_control_added(field)
            self._on_control_added(remove_button)

        self._update_remove_buttons()
        self._refresh_container_width()
        self.scroll.horizontalScrollBar().setValue(self.scroll.horizontalScrollBar().maximum())

    def remove_item(self, first_field: QLineEdit) -> None:
        if len(self.entries) <= self._min_items:
            for entry in self.entries:
                if entry.fields and entry.fields[0] is first_field:
                    for field in entry.fields:
                        field.clear()
                    break
            return

        for idx, entry in enumerate(self.entries):
            if not entry.fields or entry.fields[0] is not first_field:
                continue
            self.entries.pop(idx)
            self.fields_layout.removeWidget(entry.widget)
            if self._on_control_removed is not None:
                for field in entry.fields:
                    self._on_control_removed(field)
                self._on_control_removed(entry.remove_button)
            entry.widget.deleteLater()
            break

        self._update_remove_buttons()
        self._refresh_container_width()

    def reset(self, values: tuple[tuple[str, ...], ...]) -> None:
        while len(self.entries) > 0:
            self.remove_item(self.entries[-1].fields[0])
            if len(self.entries) == self._min_items:
                break

        if not values:
            self._ensure_min_items()
            return

        while self.entries:
            entry = self.entries[-1]
            self.entries.pop()
            self.fields_layout.removeWidget(entry.widget)
            if self._on_control_removed is not None:
                for field in entry.fields:
                    self._on_control_removed(field)
                self._on_control_removed(entry.remove_button)
            entry.widget.deleteLater()

        for row_values in values:
            self.add_item(*row_values)
        self._ensure_min_items()

    def rows(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(field.text().strip() for field in entry.fields) for entry in self.entries)

    def _ensure_min_items(self) -> None:
        while len(self.entries) < self._min_items:
            self.add_item()
        self._update_remove_buttons()
        self._refresh_container_width()
        self.scroll.horizontalScrollBar().setValue(0)

    def _normalize_values(self, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            return tuple("" for _ in self._placeholders)
        if len(values) != len(self._placeholders):
            raise ValueError("Количество значений не совпадает с количеством полей ряда")
        return tuple(values)

    def _entry_width(self) -> int:
        fields_total = sum(self._field_widths)
        separators_count = max(0, len(self._placeholders) - 1) if self._separator_text is not None else 0
        separators_total = separators_count * self._row_control_spacing * 3
        gaps = (len(self._placeholders)) * self._row_control_spacing
        return fields_total + separators_total + self._control_button_size + gaps

    def _update_remove_buttons(self) -> None:
        disable_remove = len(self.entries) <= self._min_items
        for entry in self.entries:
            entry.remove_button.setDisabled(disable_remove)

    def _refresh_container_width(self) -> None:
        self.container.adjustSize()
        width = self.container.sizeHint().width()
        height = self.container.sizeHint().height()
        self.container.resize(width, max(height, self._container_min_height))
