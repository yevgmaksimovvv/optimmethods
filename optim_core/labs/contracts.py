"""Контракты плагинов лабораторных работ для единого GUI-shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
else:
    QWidget = Any


class LabPlugin(Protocol):
    """Контракт расширения shell новой лабораторной работой."""

    @property
    def lab_id(self) -> str:
        ...

    @property
    def title(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def supports_embedded(self) -> bool:
        ...

    @property
    def supports_standalone(self) -> bool:
        ...

    def build_page(self) -> QWidget:
        ...

    def launch_standalone(self) -> None:
        ...


@dataclass(frozen=True)
class LabItemView:
    """DTO для вывода плагина в навигации shell."""

    lab_id: str
    title: str
    description: str
    supports_embedded: bool
    supports_standalone: bool
