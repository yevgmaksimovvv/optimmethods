"""Главная точка входа для запуска приложения как пакета.

Модуль нужен как тонкий bootstrap-слой:
- включает логирование;
- пробует импортировать GUI;
- формирует понятное сообщение, если не хватает зависимостей.
"""

import logging
import sys

from lr1.infrastructure.logging import configure_logging, get_log_file_path


def main() -> None:
    """Запускает GUI-приложение или завершает процесс с понятной ошибкой."""
    configure_logging()
    logger = logging.getLogger("lr1.entry")
    logger.info("Entry point main start log_file=%s", get_log_file_path())
    try:
        from lr1.ui.window import main as gui_main
    except ModuleNotFoundError as exc:
        logger.exception("GUI dependency import failed: %s", exc)
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
