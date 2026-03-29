"""Единый GUI-shell для запуска лабораторных работ как плагинов."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from optim_core.labs.plugins import build_default_registry
from optim_core.labs.registry import LabRegistry

APP_TITLE = "Методы оптимизации — лабораторные работы"


@dataclass
class _PageSlot:
    """Связка id плагина и индекса страницы в `QStackedWidget`."""

    lab_id: str
    stack_index: int
    loaded: bool


class LabsShellWindow(QMainWindow):
    """Главное окно shell c навигацией и ленивой загрузкой lab-плагинов."""

    def __init__(self, registry: LabRegistry | None = None) -> None:
        super().__init__()
        self.registry = registry or build_default_registry()
        self.setWindowTitle(APP_TITLE)
        self.resize(1680, 980)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        nav_panel = QWidget()
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(8)

        self.nav_list = QListWidget()
        self.nav_list.itemSelectionChanged.connect(self._on_lab_selected)
        nav_layout.addWidget(self.nav_list, 1)

        self.open_external_button = QPushButton("Открыть в отдельном окне")
        self.open_external_button.clicked.connect(self._open_selected_standalone)
        nav_layout.addWidget(self.open_external_button)

        root_layout.addWidget(nav_panel, 0)

        self.page_stack = QStackedWidget()
        root_layout.addWidget(self.page_stack, 1)

        self._slots: dict[str, _PageSlot] = {}
        self._populate_nav()

    def _populate_nav(self) -> None:
        for item in self.registry.items():
            list_item = QListWidgetItem(f"{item.title} — {item.description}")
            list_item.setData(Qt.UserRole, item.lab_id)
            self.nav_list.addItem(list_item)

            placeholder = QWidget()
            layout = QVBoxLayout(placeholder)
            layout.setContentsMargins(24, 24, 24, 24)
            hint = QLabel(f"Выбрана {item.title}. Страница загружается по требованию.")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)
            index = self.page_stack.addWidget(placeholder)
            self._slots[item.lab_id] = _PageSlot(lab_id=item.lab_id, stack_index=index, loaded=False)

        if self.nav_list.count() > 0:
            self.nav_list.setCurrentRow(0)

    def _on_lab_selected(self) -> None:
        item = self.nav_list.currentItem()
        if item is None:
            return
        lab_id = str(item.data(Qt.UserRole))
        slot = self._slots[lab_id]
        plugin = self.registry.get(lab_id)

        if not slot.loaded and plugin.supports_embedded:
            page = plugin.build_page()
            self.page_stack.removeWidget(self.page_stack.widget(slot.stack_index))
            self.page_stack.insertWidget(slot.stack_index, page)
            self._slots[lab_id] = _PageSlot(lab_id=lab_id, stack_index=slot.stack_index, loaded=True)

        self.page_stack.setCurrentIndex(slot.stack_index)
        self.open_external_button.setEnabled(plugin.supports_standalone)

    def _open_selected_standalone(self) -> None:
        item = self.nav_list.currentItem()
        if item is None:
            return
        lab_id = str(item.data(Qt.UserRole))
        plugin = self.registry.get(lab_id)
        if not plugin.supports_standalone:
            QMessageBox.information(self, "Недоступно", "Для этой лабораторной нет standalone-режима.")
            return
        try:
            plugin.launch_standalone()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка запуска", str(exc))


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = LabsShellWindow()
    window.show()
    app.exec()
