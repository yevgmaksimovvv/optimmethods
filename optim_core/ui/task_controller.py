"""Общий контроллер фоновых задач для Qt-UI."""

from __future__ import annotations

import logging
import traceback
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger("optim_core.ui.task_controller")


class _Worker(QObject):
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, task: Callable[[], object], label: str):
        super().__init__()
        self._task = task
        self._label = label

    def run(self) -> None:
        logger.info("worker start label=%s", self._label)
        try:
            result = self._task()
        except Exception as exc:  # noqa: BLE001
            logger.exception("worker failed label=%s error=%s", self._label, exc)
            self.failed.emit(str(exc), traceback.format_exc())
            return
        logger.info("worker done label=%s", self._label)
        self.finished.emit(result)


class TaskController(QObject):
    """Управляет жизненным циклом одной фоновой задачи."""

    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _Worker | None = None

    def is_running(self) -> bool:
        thread = self._thread
        if thread is None:
            return False
        try:
            return thread.isRunning()
        except RuntimeError:
            return False

    def start(self, label: str, task: Callable[[], object]) -> bool:
        if self.is_running():
            return False

        thread = QThread(self)
        worker = _Worker(task, label)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self.succeeded)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(self.failed)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_refs(thread, worker))

        self._thread = thread
        self._worker = worker
        thread.start()
        return True

    def _clear_refs(self, thread: QThread, worker: _Worker) -> None:
        if self._thread is thread:
            self._thread = None
        if self._worker is worker:
            self._worker = None
