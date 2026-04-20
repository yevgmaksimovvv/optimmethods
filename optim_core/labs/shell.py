"""Единый GUI-shell для встроенных лабораторных работ."""

from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget

APP_TITLE = "Методы оптимизации — лабораторные работы"


@dataclass(frozen=True)
class _LabSpec:
    """Статическое описание одной встроенной лабораторной."""

    lab_id: str
    title: str
    description: str
    window_factory: Callable[[], QMainWindow] | None = None
    standalone_module: str | None = None

    @property
    def supports_embedded(self) -> bool:
        return self.window_factory is not None

    @property
    def supports_standalone(self) -> bool:
        return self.standalone_module is not None

    def build_page(self) -> QWidget:
        if self.window_factory is None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(18, 18, 18, 18)
            hint = QLabel("Эта лабораторная работа доступна только в отдельном окне.")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)
            return page
        return _embed_main_window(self.window_factory())

    def launch_standalone(self) -> None:
        if self.standalone_module is None:
            raise RuntimeError(f"Лабораторная {self.lab_id} не поддерживает standalone-запуск")
        _launch_module(self.standalone_module)


@dataclass
class _PageSlot:
    """Связка id лабораторной и индекса страницы в `QStackedWidget`."""

    lab_id: str
    stack_index: int
    loaded: bool


def _embed_main_window(window: QMainWindow) -> QWidget:
    content = window.takeCentralWidget()
    if content is None:
        raise RuntimeError("Окно не содержит central widget и не может быть встроено")

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(content)
    return page


def _launch_module(module_name: str) -> None:
    subprocess.Popen([sys.executable, "-m", module_name])


def _build_lr1_window() -> QMainWindow:
    from lr1.ui.window import ExtremumWindow

    return ExtremumWindow()


def _build_lr2_window() -> QMainWindow:
    from lr2.ui.window import RosenbrockWindow

    return RosenbrockWindow()


def _build_lr3_window() -> QMainWindow:
    from lr3.ui.window import GradientMethodsWindow

    return GradientMethodsWindow()


def _build_lr5_window() -> QMainWindow:
    from lr5.ui.window import BarrierWindow

    return BarrierWindow()


DEFAULT_LABS: tuple[_LabSpec, ...] = (
    _LabSpec(
        lab_id="lr1",
        title="ЛР1",
        description="Методы одномерной оптимизации",
        window_factory=_build_lr1_window,
        standalone_module="lr1",
    ),
    _LabSpec(
        lab_id="lr2",
        title="ЛР2",
        description="Метод Розенброка (непрерывный и дискретный шаг)",
        window_factory=_build_lr2_window,
        standalone_module="lr2",
    ),
    _LabSpec(
        lab_id="lr3",
        title="ЛР3",
        description="Градиентные методы",
        window_factory=_build_lr3_window,
        standalone_module="lr3",
    ),
    _LabSpec(
        lab_id="lr5",
        title="ЛР5",
        description="Метод барьерных функций",
        window_factory=_build_lr5_window,
        standalone_module="lr5",
    ),
)


class LabsShellWindow(QMainWindow):
    """Главное окно shell c навигацией и ленивой загрузкой страниц лабораторных."""

    def __init__(self) -> None:
        super().__init__()
        self.labs = DEFAULT_LABS
        self._labs_by_id = {lab.lab_id: lab for lab in self.labs}
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
        for lab in self.labs:
            list_item = QListWidgetItem(f"{lab.title} — {lab.description}")
            list_item.setData(Qt.UserRole, lab.lab_id)
            self.nav_list.addItem(list_item)

            placeholder = QWidget()
            layout = QVBoxLayout(placeholder)
            layout.setContentsMargins(24, 24, 24, 24)
            hint = QLabel(f"Выбрана {lab.title}. Страница загружается по требованию.")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)
            index = self.page_stack.addWidget(placeholder)
            self._slots[lab.lab_id] = _PageSlot(lab_id=lab.lab_id, stack_index=index, loaded=False)

        if self.nav_list.count() > 0:
            self.nav_list.setCurrentRow(0)

    def _on_lab_selected(self) -> None:
        item = self.nav_list.currentItem()
        if item is None:
            return
        lab_id = str(item.data(Qt.UserRole))
        slot = self._slots[lab_id]
        lab = self._labs_by_id[lab_id]

        if not slot.loaded and lab.supports_embedded:
            page = lab.build_page()
            self.page_stack.removeWidget(self.page_stack.widget(slot.stack_index))
            self.page_stack.insertWidget(slot.stack_index, page)
            self._slots[lab_id] = _PageSlot(lab_id=lab_id, stack_index=slot.stack_index, loaded=True)

        self.page_stack.setCurrentIndex(slot.stack_index)
        self.open_external_button.setEnabled(lab.supports_standalone)

    def _open_selected_standalone(self) -> None:
        item = self.nav_list.currentItem()
        if item is None:
            return
        lab_id = str(item.data(Qt.UserRole))
        lab = self._labs_by_id[lab_id]
        if not lab.supports_standalone:
            QMessageBox.information(self, "Недоступно", "Для этой лабораторной нет standalone-режима.")
            return
        try:
            lab.launch_standalone()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка запуска", str(exc))


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = LabsShellWindow()
    window.show()
    app.exec()
