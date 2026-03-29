"""Главная точка входа для запуска приложения как пакета.

Модуль нужен как тонкий bootstrap-слой:
- включает логирование;
- пробует импортировать GUI;
- формирует понятное сообщение, если не хватает зависимостей.
"""

import logging

from optim_core.entrypoint import GuiEntryPointSpec, run_gui_entry

from lr1.infrastructure.logging import configure_logging, get_log_file_path


def main() -> None:
    """Запускает GUI-приложение или завершает процесс с понятной ошибкой."""
    configure_logging()
    logger = logging.getLogger("lr1.entry")
    logger.info("Entry point main start log_file=%s", get_log_file_path())
    run_gui_entry(
        GuiEntryPointSpec(
            gui_import_path="lr1.ui.window:main",
            requirements_hint_path="optimmethods/requirements.txt",
            logger_name="lr1.entry",
        )
    )


if __name__ == "__main__":
    main()
