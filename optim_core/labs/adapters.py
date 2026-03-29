"""Адаптеры для подключения существующих GUI к shell-плагинам."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget


class EmbeddedMainWindowPage(QWidget):
    """Встраивает центральный виджет существующего `QMainWindow` в страницу shell."""

    def __init__(self, window: QMainWindow) -> None:
        super().__init__()
        self._window = window
        content = window.takeCentralWidget()
        if content is None:
            raise RuntimeError("Окно не содержит central widget и не может быть встроено")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(content)
        self._content = content


@dataclass(frozen=True)
class QtMainWindowEmbedAdapter:
    """Фабрика адаптера `QMainWindow -> QWidget page`."""

    window_factory: Callable[[], QMainWindow]

    def build_page(self) -> QWidget:
        return EmbeddedMainWindowPage(self.window_factory())
