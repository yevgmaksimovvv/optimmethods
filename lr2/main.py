"""Главная точка входа для запуска ЛР2 как пакета.

Тонкий bootstrap-слой:
- пробует импортировать GUI;
- формирует понятное сообщение, если не хватает зависимостей.
"""

from __future__ import annotations

from optim_core.entrypoint import GuiEntryPointSpec, run_gui_entry


def main() -> None:
    """Запускает GUI ЛР2 или завершает процесс с понятной ошибкой."""
    run_gui_entry(
        GuiEntryPointSpec(
            gui_import_path="lr2.ui.window:main",
            requirements_hint_path="optimmethods/requirements.txt",
        )
    )


if __name__ == "__main__":
    main()
