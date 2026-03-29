"""Общий bootstrap для GUI-точек входа."""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GuiEntryPointSpec:
    """Контракт запуска GUI из import path формата 'module.submodule:callable'."""

    gui_import_path: str
    requirements_hint_path: str
    logger_name: str | None = None


def run_gui_entry(spec: GuiEntryPointSpec) -> None:
    """Импортирует и запускает GUI-функцию, либо завершает процесс с понятной ошибкой."""
    logger = logging.getLogger(spec.logger_name) if spec.logger_name else None
    gui_main = _load_gui_main(spec.gui_import_path, logger, spec.requirements_hint_path)
    gui_main()


def _load_gui_main(
    gui_import_path: str,
    logger: logging.Logger | None,
    requirements_hint_path: str,
) -> Callable[[], None]:
    module_name, attr_name = _split_import_path(gui_import_path)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if logger is not None:
            logger.exception("GUI dependency import failed: %s", exc)
        dependency = exc.name or "unknown"
        raise SystemExit(
            "Не хватает зависимости для GUI: "
            f"{dependency}\n"
            "Установи зависимости командой:\n"
            f"{sys.executable} -m pip install -r {requirements_hint_path}"
        ) from exc

    gui_main = getattr(module, attr_name, None)
    if not callable(gui_main):
        raise RuntimeError(f"Точка входа '{gui_import_path}' должна быть вызываемой функцией без аргументов.")
    return gui_main


def _split_import_path(gui_import_path: str) -> tuple[str, str]:
    parts = gui_import_path.split(":", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Неверный gui_import_path: ожидается формат 'module.submodule:callable'.")
    return parts[0], parts[1]
