import logging
import sys


if __package__:
    from .logging_setup import configure_logging, get_log_file_path
else:
    from logging_setup import configure_logging, get_log_file_path


def main() -> None:
    configure_logging()
    logger = logging.getLogger("lr1.entry")
    logger.info("Entry point main start log_file=%s", get_log_file_path())
    try:
        if __package__:
            from .gui_app import main as gui_main
        else:
            from gui_app import main as gui_main
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
