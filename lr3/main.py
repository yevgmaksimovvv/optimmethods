"""Точка входа GUI для ЛР3."""

from __future__ import annotations

from optim_core.entrypoint import GuiEntryPointSpec, run_gui_entry


def main() -> None:
    """Запускает GUI ЛР3 с явной ошибкой при отсутствии зависимостей."""
    run_gui_entry(
        GuiEntryPointSpec(
            gui_import_path="lr3.ui.window:main",
            requirements_hint_path="requirements.txt",
            logger_name="lr3.entry",
        )
    )


if __name__ == "__main__":
    main()
