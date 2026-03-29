"""Главная точка входа для запуска ЛР2 как пакета.

Тонкий bootstrap-слой:
- пробует импортировать GUI;
- формирует понятное сообщение, если не хватает зависимостей.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Запускает GUI ЛР2 или завершает процесс с понятной ошибкой."""
    try:
        from lr2.ui.window import main as gui_main
    except ModuleNotFoundError as exc:
        dependency = exc.name or "unknown"
        raise SystemExit(
            "Не хватает зависимости для GUI: "
            f"{dependency}\n"
            "Установи зависимости командой:\n"
            f"{sys.executable} -m pip install -r optimmethods/requirements.txt"
        ) from exc

    gui_main()


if __name__ == "__main__":
    main()
