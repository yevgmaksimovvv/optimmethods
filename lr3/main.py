"""Точка входа GUI для ЛР3."""

from __future__ import annotations

import logging
import sys


def main() -> None:
    """Запускает GUI ЛР3 с явной ошибкой при отсутствии зависимостей."""
    logger = logging.getLogger("lr3.entry")
    try:
        from lr3.ui.window import main as gui_main
    except ModuleNotFoundError as exc:
        logger.exception("GUI dependency import failed: %s", exc)
        dependency = exc.name or "unknown"
        raise SystemExit(
            "Не хватает зависимости для GUI: "
            f"{dependency}\n"
            "Установи зависимости командой:\n"
            f"{sys.executable} -m pip install -r requirements.txt"
        ) from exc

    gui_main()


if __name__ == "__main__":
    main()
