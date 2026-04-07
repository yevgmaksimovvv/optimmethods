"""Стандартные плагины лабораторных работ и сборка реестра."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from optim_core.labs.registry import LabRegistry

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
else:
    QWidget = Any


@dataclass(frozen=True)
class DefaultLabPlugin:
    """Базовая реализация lab-плагина shell."""

    lab_id: str
    title: str
    description: str
    build_page_fn: Callable[[], QWidget] | None = None
    standalone_launcher: Callable[[], None] | None = None

    @property
    def supports_embedded(self) -> bool:
        return self.build_page_fn is not None

    @property
    def supports_standalone(self) -> bool:
        return self.standalone_launcher is not None

    def build_page(self) -> QWidget:
        if self.build_page_fn is None:
            from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(18, 18, 18, 18)
            hint = QLabel("Эта лабораторная работа доступна только в отдельном окне.")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)
            return page
        return self.build_page_fn()

    def launch_standalone(self) -> None:
        if self.standalone_launcher is None:
            raise RuntimeError(f"Плагин {self.lab_id} не поддерживает standalone-запуск")
        self.standalone_launcher()


def _launch_module(module_name: str) -> None:
    subprocess.Popen([sys.executable, "-m", module_name])


def _build_lr1_page() -> QWidget:
    from optim_core.labs.adapters import QtMainWindowEmbedAdapter

    from lr1.ui.window import ExtremumWindow

    return QtMainWindowEmbedAdapter(window_factory=ExtremumWindow).build_page()


def _build_lr2_page() -> QWidget:
    from optim_core.labs.adapters import QtMainWindowEmbedAdapter

    from lr2.ui.window import RosenbrockWindow

    return QtMainWindowEmbedAdapter(window_factory=RosenbrockWindow).build_page()


def _build_lr3_page() -> QWidget:
    from optim_core.labs.adapters import QtMainWindowEmbedAdapter

    from lr3.ui.window import GradientMethodsWindow

    return QtMainWindowEmbedAdapter(window_factory=GradientMethodsWindow).build_page()


def build_default_registry() -> LabRegistry:
    """Собирает реестр доступных лабораторных по умолчанию."""
    registry = LabRegistry()
    registry.register(
        DefaultLabPlugin(
            lab_id="lr1",
            title="ЛР1",
            description="Методы одномерной оптимизации",
            build_page_fn=_build_lr1_page,
            standalone_launcher=lambda: _launch_module("lr1"),
        )
    )
    registry.register(
        DefaultLabPlugin(
            lab_id="lr2",
            title="ЛР2",
            description="Метод Розенброка (непрерывный и дискретный шаг)",
            build_page_fn=_build_lr2_page,
            standalone_launcher=lambda: _launch_module("lr2"),
        )
    )
    registry.register(
        DefaultLabPlugin(
            lab_id="lr3",
            title="ЛР3",
            description="Градиентные методы",
            build_page_fn=_build_lr3_page,
            standalone_launcher=lambda: _launch_module("lr3"),
        )
    )
    return registry
