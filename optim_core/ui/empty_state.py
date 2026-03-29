"""Переиспользуемые empty-state страницы для вкладок/панелей."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget


def create_centered_empty_state_page(
    *,
    title: str,
    description: str,
    hint: str,
    card_object_name: str = "SummaryEmptyCard",
    title_object_name: str = "SummaryEmptyTitle",
    description_object_name: str = "SummaryEmptyText",
    hint_object_name: str = "SectionHint",
    max_card_width: int = 700,
) -> tuple[QWidget, QLabel]:
    """Создаёт страницу с центрированной карточкой пустого состояния."""
    page = QWidget()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(16, 24, 16, 0)
    page_layout.setSpacing(0)

    card = QWidget()
    card.setObjectName(card_object_name)
    card.setMaximumWidth(max_card_width)
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(30, 24, 30, 24)
    card_layout.setSpacing(18)

    title_label = QLabel(title)
    title_label.setObjectName(title_object_name)
    title_label.setAlignment(Qt.AlignCenter)
    card_layout.addWidget(title_label)

    description_label = QLabel(description)
    description_label.setObjectName(description_object_name)
    description_label.setAlignment(Qt.AlignCenter)
    description_label.setWordWrap(True)
    card_layout.addWidget(description_label)

    hint_label = QLabel(hint)
    hint_label.setObjectName(hint_object_name)
    hint_label.setAlignment(Qt.AlignCenter)
    hint_label.setWordWrap(True)
    card_layout.addWidget(hint_label)

    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(card, 1)
    row.addStretch(1)
    page_layout.addLayout(row)
    page_layout.addStretch(1)
    return page, hint_label


class EmptyStateStack(QStackedWidget):
    """Двухстраничный стек: `empty-state` и контент."""

    def __init__(
        self,
        *,
        title: str,
        description: str,
        hint: str,
        content_widget: QWidget,
        card_object_name: str = "SummaryEmptyCard",
        title_object_name: str = "SummaryEmptyTitle",
        description_object_name: str = "SummaryEmptyText",
        hint_object_name: str = "SectionHint",
        max_card_width: int = 700,
    ) -> None:
        super().__init__()
        empty_page, hint_label = create_centered_empty_state_page(
            title=title,
            description=description,
            hint=hint,
            card_object_name=card_object_name,
            title_object_name=title_object_name,
            description_object_name=description_object_name,
            hint_object_name=hint_object_name,
            max_card_width=max_card_width,
        )
        self.hint_label = hint_label
        self.addWidget(empty_page)
        self.addWidget(content_widget)

    def set_empty(self, is_empty: bool) -> None:
        self.setCurrentIndex(0 if is_empty else 1)
