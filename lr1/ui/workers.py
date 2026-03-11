"""Фоновые задачи и виджет холста для отображения графиков.

UI не должен блокироваться во время расчётов и рендеринга фигур, поэтому
здесь лежит минимальная обвязка над `QThread`. Отдельно вынесен `PlotCanvas`,
который аккуратно работает внутри прокручиваемой области интерфейса.
"""

import logging
import traceback
from typing import Callable, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import QScrollArea, QSizePolicy, QWidget

from lr1.infrastructure.logging import configure_logging


configure_logging()
logger = logging.getLogger("lr1.gui_workers")


class Worker(QObject):
    """Исполнитель одной фоновой задачи внутри отдельного потока."""
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, task: Callable[[], object], label: str):
        """Сохраняет callable-задачу и её человекочитаемую метку для логов."""
        super().__init__()
        self.task = task
        self.label = label

    def run(self) -> None:
        """Запускает задачу и публикует либо результат, либо traceback ошибки."""
        logger.info("Worker start label=%s", self.label)
        try:
            result = self.task()
        except Exception as exc:
            logger.exception("Worker failed label=%s error=%s", self.label, exc)
            self.failed.emit(str(exc), traceback.format_exc())
            return
        logger.info("Worker done label=%s", self.label)
        self.finished.emit(result)


class TaskController(QObject):
    """Управляет жизненным циклом фоновой задачи и её потока.

    Класс скрывает от окна всю низкоуровневую возню с `QThread`: создание,
    подключение сигналов, очистку ссылок и защиту от повторного запуска.
    """
    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, parent: Optional[QObject] = None):
        """Создаёт контроллер без активной задачи."""
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None

    def is_running(self) -> bool:
        """Показывает, выполняется ли сейчас фоновая задача."""
        thread = self._thread
        if thread is None:
            return False
        try:
            return thread.isRunning()
        except RuntimeError:
            return False

    def start(self, label: str, task: Callable[[], object]) -> bool:
        """Запускает новую задачу, если предыдущая уже завершена.

        Возвращает `False`, когда контроллер ещё занят и повторный запуск
        нужно проигнорировать.
        """
        if self.is_running():
            return False

        thread = QThread(self)
        worker = Worker(task, label)
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

    def _clear_refs(self, thread: QThread, worker: Worker) -> None:
        """Обнуляет ссылки после завершения потока, чтобы объект можно было переиспользовать."""
        if self._thread is thread:
            self._thread = None
        if self._worker is worker:
            self._worker = None


class PlotCanvas(FigureCanvasQTAgg):
    """Matplotlib-холст, адаптированный под прокручиваемую область Qt."""
    def __init__(self, figure: Figure, parent: Optional[QWidget] = None):
        """Встраивает фигуру в Qt-виджет и настраивает политику размеров."""
        super().__init__(figure)
        self.setParent(parent)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.updateGeometry()

    def wheelEvent(self, event) -> None:
        """Прокидывает колесо мыши в родительский `QScrollArea`, если он есть.

        Без этого при наведении курсора на график начинает зумиться сам холст,
        а не прокручиваться страница с результатами, что неудобно для длинных
        отчётов.
        """
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()

        if isinstance(parent, QScrollArea):
            delta_y = event.angleDelta().y()
            if delta_y:
                scrollbar = parent.verticalScrollBar()
                step = max(scrollbar.singleStep(), 24)
                scrollbar.setValue(scrollbar.value() - int((delta_y / 120) * step * 3))
                event.accept()
                return

        super().wheelEvent(event)
