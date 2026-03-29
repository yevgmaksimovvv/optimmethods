"""Настройка файлового логирования для всего приложения.

Модуль специально вынесен отдельно от бизнес-логики:
- чтобы любое место в проекте могло безопасно вызвать `configure_logging()`;
- чтобы конфигурация логирования была единой для GUI, расчётов и вспомогательных
  сервисов;
- чтобы в лог сразу попадали необработанные исключения и ошибки из потоков.
"""

import logging
import os
import platform
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "lr1_debug.log"
_CONFIGURED = False


def get_log_file_path() -> Path:
    """Возвращает путь к файлу лога и гарантирует существование каталога."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_FILE


def configure_logging() -> Path:
    """Инициализирует общее логирование приложения один раз за процесс.

    Возвращаемое значение:
    - путь к файлу, в который будет писаться подробный debug-лог.

    Повторный вызов безопасен: если конфигурация уже была выполнена,
    функция просто вернёт тот же путь без повторного добавления handler'ов.
    """
    global _CONFIGURED

    log_path = get_log_file_path()
    if _CONFIGURED:
        return log_path

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    app_logger = logging.getLogger("lr1")
    app_logger.setLevel(logging.DEBUG)

    log_path_str = str(log_path)
    has_handler = any(
        isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", None) == log_path_str
        for handler in root_logger.handlers
    )

    if not has_handler:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)s "
                "[pid=%(process)d thread=%(threadName)s] "
                "%(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    for noisy_logger_name in (
        "matplotlib",
        "PIL",
        "urllib3",
    ):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    def log_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("lr1.unhandled").critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def log_thread_exception(args):
        logging.getLogger("lr1.threading").critical(
            "Unhandled threading exception in %s",
            args.thread.name if args.thread else "unknown-thread",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = log_uncaught_exception
    if hasattr(threading, "excepthook"):
        threading.excepthook = log_thread_exception

    logging.getLogger("lr1.bootstrap").info(
        "Logging configured. file=%s python=%s platform=%s cwd=%s pid=%s",
        log_path,
        sys.version.replace("\n", " "),
        platform.platform(),
        os.getcwd(),
        os.getpid(),
    )
    _CONFIGURED = True
    return log_path
