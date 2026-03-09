import logging
import sys
import traceback
from typing import Callable, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .app_models import AppState, GridRunResult, InputConfig, RunReport
    from .app_service import build_input_config, run_full_grid, run_single
    from .logging_setup import configure_logging, get_log_file_path
    from .plotting import build_plot_figure
    from .search_methods import METHOD_SPECS
else:
    from app_models import AppState, GridRunResult, InputConfig, RunReport
    from app_service import build_input_config, run_full_grid, run_single
    from logging_setup import configure_logging, get_log_file_path
    from plotting import build_plot_figure
    from search_methods import METHOD_SPECS


configure_logging()
logger = logging.getLogger("lr1.gui")


EPS_PRESETS = ("0.1", "0.01", "0.001")
L_PRESETS = ("0.1", "0.01")


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, task: Callable[[], object], label: str):
        super().__init__()
        self.task = task
        self.label = label

    def run(self) -> None:
        logger.info("Worker start label=%s", self.label)
        try:
            result = self.task()
        except Exception as exc:
            logger.exception("Worker failed label=%s error=%s", self.label, exc)
            self.failed.emit(str(exc), traceback.format_exc())
            return
        logger.info("Worker done label=%s", self.label)
        self.finished.emit(result)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, figure: Figure, parent: Optional[QWidget] = None):
        super().__init__(figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class ExtremumWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("ExtremumWindow init start")
        self.state = AppState()
        self.calc_thread: Optional[QThread] = None
        self.calc_worker: Optional[Worker] = None
        self.plot_thread: Optional[QThread] = None
        self.plot_worker: Optional[Worker] = None
        self.plot_canvas: Optional[PlotCanvas] = None
        self.current_plot_version = 0
        self.control_widgets: List[QWidget] = []
        self.table_method_buttons: List[QRadioButton] = []

        self.setWindowTitle("Поиск экстремума")
        self.resize(1480, 940)
        self._apply_styles()
        self._build_ui()
        self._set_defaults()
        logger.info("ExtremumWindow init done log_file=%s", get_log_file_path())

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #1e1f24;
                color: #f0f2f5;
                font-family: "SF Pro Text", "Helvetica Neue", "Arial";
                font-size: 15px;
            }
            QGroupBox {
                border: 1px solid #4b4f5c;
                border-radius: 10px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #d8dbe2;
            }
            QPushButton {
                background: #3c4358;
                border: 1px solid #59627d;
                border-radius: 8px;
                padding: 10px 14px;
                color: #f5f7fb;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #4a5470;
            }
            QPushButton:pressed {
                background: #31384a;
            }
            QPushButton:disabled {
                background: #2c2f39;
                color: #8f95a3;
                border-color: #3a3e49;
            }
            QPushButton[role="primary"] {
                background: #1f6feb;
                border-color: #2f81f7;
            }
            QPushButton[role="primary"]:hover {
                background: #2f81f7;
            }
            QPushButton[role="primary"]:pressed {
                background: #1858ba;
            }
            QLineEdit, QListWidget, QTreeWidget, QTableWidget {
                background: #14161b;
                border: 1px solid #464b59;
                border-radius: 8px;
                padding: 8px;
                color: #f5f7fb;
                selection-background-color: #2a6df4;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-radius: 6px;
            }
            QHeaderView::section {
                background: #252933;
                color: #d8dbe2;
                border: 0;
                border-right: 1px solid #3c4358;
                border-bottom: 1px solid #3c4358;
                padding: 8px 10px;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 1px solid #4b4f5c;
                border-radius: 10px;
                top: -1px;
            }
            QTabBar::tab {
                background: #2c303b;
                border: 1px solid #4b4f5c;
                padding: 10px 18px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #3b4254;
                color: #ffffff;
            }
            QLabel#StatusLabel {
                color: #b6becf;
                font-size: 13px;
                padding: 4px 0;
            }
            QLabel#SectionHint {
                color: #c8cfdb;
                font-size: 14px;
                line-height: 1.4;
                padding: 4px 2px;
            }
            QRadioButton {
                spacing: 8px;
                padding: 4px 0;
                min-height: 24px;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self._build_scenario_box(left_layout)
        self._build_parameters_box(left_layout)
        self._build_actions_box(left_layout)

        self.status_label = QLabel("Готово")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self._build_tabs(right_layout)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1080])

    def _build_scenario_box(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Сценарий")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        self.function_group = QButtonGroup(self)
        self.function_buttons = self._create_radio_row(
            grid,
            row=0,
            label="Функция",
            group=self.function_group,
            options=(("F1", "F1"), ("F2", "F2")),
            callback=self.on_function_change,
        )
        self.kind_group = QButtonGroup(self)
        self.kind_buttons = self._create_radio_row(
            grid,
            row=1,
            label="Цель",
            group=self.kind_group,
            options=(("Максимум", "max"), ("Минимум", "min")),
        )
        self.method_group = QButtonGroup(self)
        self.method_buttons = self._create_radio_grid(
            grid,
            row=2,
            label="Алгоритм",
            group=self.method_group,
            options=(
                ("Все", "all"),
                ("Дихотомия", "dichotomy"),
                ("Золотое сечение", "golden"),
                ("Фибоначчи", "fibonacci"),
            ),
            columns=2,
        )
        layout.addWidget(box)

    def _build_parameters_box(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Параметры")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        self.a_input = self._add_labeled_input(grid, 0, "Левая граница a")
        self.b_input = self._add_labeled_input(grid, 1, "Правая граница b")
        self.eps_input = self._add_labeled_input(grid, 2, "Точность ε")
        self._add_quick_buttons(grid, 3, EPS_PRESETS, self.eps_input)
        self.l_input = self._add_labeled_input(grid, 4, "Длина интервала l")
        self._add_quick_buttons(grid, 5, L_PRESETS, self.l_input)

        layout.addWidget(box)

    def _build_actions_box(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Действия")
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(8)

        self.run_button = QPushButton("Один расчёт")
        self.run_button.clicked.connect(self.handle_run_single)
        self.run_button.setProperty("role", "primary")
        box_layout.addWidget(self.run_button)

        self.grid_button = QPushButton("Исследование ε × l")
        self.grid_button.clicked.connect(self.handle_run_full_grid)
        box_layout.addWidget(self.grid_button)

        self.plot_button = QPushButton("Обновить график")
        self.plot_button.clicked.connect(self.render_last_plot)
        self.plot_button.setEnabled(False)
        box_layout.addWidget(self.plot_button)

        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self.clear_output)
        box_layout.addWidget(self.clear_button)

        layout.addWidget(box)

        self.control_widgets.extend(
            [
                self.a_input,
                self.b_input,
                self.eps_input,
                self.l_input,
                self.run_button,
                self.grid_button,
                self.plot_button,
                self.clear_button,
                *self.function_buttons,
                *self.kind_buttons,
                *self.method_buttons,
            ]
        )

    def _build_tabs(self, layout: QVBoxLayout) -> None:
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        summary_tab = QWidget()
        summary_tab_layout = QVBoxLayout(summary_tab)
        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setFrameShape(QScrollArea.NoFrame)
        summary_content = QWidget()
        summary_layout = QVBoxLayout(summary_content)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(12)

        self.summary_context_box = QGroupBox("Контекст")
        context_layout = QVBoxLayout(self.summary_context_box)
        self.summary_context_table = self._create_table_widget()
        context_layout.addWidget(self.summary_context_table)
        summary_layout.addWidget(self.summary_context_box)

        self.summary_results_box = QGroupBox("Результаты")
        results_layout = QVBoxLayout(self.summary_results_box)
        self.summary_results_table = self._create_table_widget()
        results_layout.addWidget(self.summary_results_table)
        summary_layout.addWidget(self.summary_results_box)

        self.summary_reference_box = QGroupBox("Теоретический ориентир")
        reference_layout = QVBoxLayout(self.summary_reference_box)
        self.summary_reference_table = self._create_table_widget()
        self.summary_analytic_label = QLabel("—")
        self.summary_analytic_label.setObjectName("SectionHint")
        self.summary_analytic_label.setWordWrap(True)
        reference_layout.addWidget(self.summary_reference_table)
        reference_layout.addWidget(self.summary_analytic_label)
        summary_layout.addWidget(self.summary_reference_box)

        self.summary_skipped_box = QGroupBox("Пропущенные наборы параметров")
        skipped_layout = QVBoxLayout(self.summary_skipped_box)
        self.summary_skipped_table = self._create_table_widget()
        skipped_layout.addWidget(self.summary_skipped_table)
        summary_layout.addWidget(self.summary_skipped_box)

        self.summary_notes_box = QGroupBox("Выводы")
        notes_layout = QVBoxLayout(self.summary_notes_box)
        self.summary_notes_list = QListWidget()
        self.summary_notes_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.summary_notes_list.setFocusPolicy(Qt.NoFocus)
        notes_layout.addWidget(self.summary_notes_list)
        summary_layout.addWidget(self.summary_notes_box)
        summary_layout.addStretch(1)

        summary_scroll.setWidget(summary_content)
        summary_tab_layout.addWidget(summary_scroll)
        self.tabs.addTab(summary_tab, "Сводка")

        iterations_tab = QWidget()
        iterations_layout = QVBoxLayout(iterations_tab)
        self.table_method_box = QGroupBox("Итерации выбранного метода")
        method_layout = QHBoxLayout(self.table_method_box)
        method_layout.setContentsMargins(10, 10, 10, 10)
        method_layout.setSpacing(10)
        self.table_method_buttons_layout = method_layout
        iterations_layout.addWidget(self.table_method_box)

        self.iterations_tree = QTreeWidget()
        self.iterations_tree.setColumnCount(7)
        self.iterations_tree.setHeaderLabels(("k", "a_k", "b_k", "lambda_k", "mu_k", "F(lambda_k)", "F(mu_k)"))
        self.iterations_tree.setAlternatingRowColors(True)
        self.iterations_tree.setRootIsDecorated(False)
        self.grid_run_list = QListWidget()
        self.grid_run_list.hide()
        self.grid_run_list.currentRowChanged.connect(self.on_grid_run_change)
        iterations_layout.addWidget(self.grid_run_list)
        iterations_layout.addWidget(self.iterations_tree)
        self.tabs.addTab(iterations_tab, "Итерации")

        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.plot_state_label = QLabel("График появится после расчёта.")
        self.plot_state_label.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.plot_state_label)
        self.plot_host = QWidget()
        self.plot_host_layout = QVBoxLayout(self.plot_host)
        self.plot_host_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot_host)
        self.tabs.addTab(plot_tab, "График")
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def _create_radio_row(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        group: QButtonGroup,
        options,
        callback: Optional[Callable[[], None]] = None,
    ) -> List[QRadioButton]:
        grid.addWidget(QLabel(label), row, 0)
        holder = QWidget()
        holder_layout = QHBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(18)
        buttons: List[QRadioButton] = []
        for caption, value in options:
            button = QRadioButton(caption)
            button.setProperty("choice_value", value)
            group.addButton(button)
            holder_layout.addWidget(button)
            buttons.append(button)
            if callback is not None:
                button.toggled.connect(callback)
        holder_layout.addStretch(1)
        grid.addWidget(holder, row, 1, 1, 3)
        return buttons

    def _create_radio_grid(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        group: QButtonGroup,
        options,
        columns: int,
    ) -> List[QRadioButton]:
        grid.addWidget(QLabel(label), row, 0, Qt.AlignTop)
        holder = QWidget()
        holder_layout = QGridLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setHorizontalSpacing(18)
        holder_layout.setVerticalSpacing(8)
        buttons: List[QRadioButton] = []
        for index, (caption, value) in enumerate(options):
            button = QRadioButton(caption)
            button.setProperty("choice_value", value)
            group.addButton(button)
            holder_layout.addWidget(button, index // columns, index % columns)
            buttons.append(button)
        grid.addWidget(holder, row, 1, 1, 3)
        return buttons

    def _add_labeled_input(self, grid: QGridLayout, row: int, label: str) -> QLineEdit:
        grid.addWidget(QLabel(label), row, 0)
        line = QLineEdit()
        grid.addWidget(line, row, 1, 1, 2)
        return line

    def _add_quick_buttons(self, grid: QGridLayout, row: int, values, target: QLineEdit) -> None:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for value in values:
            button = QPushButton(value)
            button.clicked.connect(lambda checked=False, value=value: target.setText(value))
            layout.addWidget(button)
            self.control_widgets.append(button)
        layout.addStretch(1)
        grid.addWidget(holder, row, 1, 1, 2)

    def _create_table_widget(self) -> QTableWidget:
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setHighlightSections(False)
        table.setWordWrap(True)
        return table

    def _set_table_data(self, table: QTableWidget, headers: List[str], rows: List[List[str]]) -> None:
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                table.setItem(row_index, col_index, item)
        header = table.horizontalHeader()
        for index in range(len(headers)):
            header.setSectionResizeMode(index, QHeaderView.ResizeToContents)
        if headers:
            header.setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)
        table.resizeRowsToContents()

    def _format_interval(self, interval) -> str:
        return f"[{interval[0]:.6f}, {interval[1]:.6f}]"

    def _format_optional_float(self, value: Optional[float]) -> str:
        return "—" if value is None else f"{value:.10f}"

    def _result_error_pair(self, report: RunReport, x_value: float, f_value: float) -> List[str]:
        if report.reference_x is None or report.reference_f is None:
            return ["—", "—"]
        return [
            f"{abs(x_value - report.reference_x):.10f}",
            f"{abs(f_value - report.reference_f):.10f}",
        ]

    def _context_rows(self, report: RunReport) -> List[List[str]]:
        rows = [
            ["Функция", report.function_spec.title],
            ["Режим", "Полное исследование" if report.mode == "grid" else "Один расчёт"],
            ["Поиск", report.kind],
            ["Исходный интервал", self._format_interval(report.interval_raw)],
            ["Интервал вычислений", self._format_interval(report.interval)],
        ]
        if report.mode == "single":
            rows.extend(
                [
                    ["Точность ε", self._format_optional_float(report.eps)],
                    ["l", self._format_optional_float(report.l)],
                ]
            )
        else:
            rows.extend(
                [
                    ["Значения ε", ", ".join(f"{value:g}" for value in (0.1, 0.01, 0.001))],
                    ["Сетка l", ", ".join(f"{value:g}" for value in (0.1, 0.01))],
                ]
            )
        return rows

    def _single_result_rows(self, report: RunReport) -> List[List[str]]:
        rows: List[List[str]] = []
        for method_key in report.method_keys:
            result = report.results_by_method.get(method_key)
            if result is None:
                continue
            dx, df = self._result_error_pair(report, result.x_opt, result.f_opt)
            rows.append(
                [
                    result.method,
                    f"{result.x_opt:.10f}",
                    f"{result.f_opt:.10f}",
                    str(len(result.iterations)),
                    str(result.func_evals),
                    self._format_interval(result.interval_final),
                    dx,
                    df,
                ]
            )
        return rows

    def _grid_result_rows(self, report: RunReport) -> List[List[str]]:
        rows: List[List[str]] = []
        for method_key in report.method_keys:
            for run in report.grid_runs_by_method.get(method_key, ()):
                dx, df = self._result_error_pair(report, run.result.x_opt, run.result.f_opt)
                rows.append(
                [
                    run.result.method,
                    f"{run.eps:g}",
                    f"{run.l:g}",
                    f"{run.result.x_opt:.10f}",
                    f"{run.result.f_opt:.10f}",
                        str(len(run.result.iterations)),
                        str(run.result.func_evals),
                        self._format_interval(run.result.interval_final),
                        dx,
                        df,
                    ]
                )
        return rows

    def _populate_summary(self) -> None:
        report = self.state.last_report
        if report is None:
            self._set_table_data(self.summary_context_table, ["Параметр", "Значение"], [])
            self._set_table_data(self.summary_results_table, ["Результат"], [])
            self._set_table_data(self.summary_reference_table, ["Параметр", "Значение"], [])
            self._set_table_data(self.summary_skipped_table, ["Метод", "ε", "l", "Причина"], [])
            self.summary_skipped_box.hide()
            self.summary_notes_box.hide()
            self.summary_reference_box.hide()
            self.summary_analytic_label.setText("—")
            self.summary_notes_list.clear()
            return

        self._set_table_data(self.summary_context_table, ["Параметр", "Значение"], self._context_rows(report))

        if report.mode == "grid":
            result_headers = [
                "Метод",
                "ε",
                "l",
                "x*",
                "f(x*)",
                "Итерации",
                "Вызовы функции",
                "Финальный интервал",
                "|dx|",
                "|df|",
            ]
            result_rows = self._grid_result_rows(report)
        else:
            result_headers = [
                "Метод",
                "x*",
                "f(x*)",
                "Итерации",
                "Вызовы функции",
                "Финальный интервал",
                "|dx|",
                "|df|",
            ]
            result_rows = self._single_result_rows(report)
        self._set_table_data(self.summary_results_table, result_headers, result_rows)

        reference_rows = [
            ["x*", self._format_optional_float(report.reference_x)],
            ["f(x*)", self._format_optional_float(report.reference_f)],
            ["Источник", report.reference_source or "—"],
        ]
        self._set_table_data(self.summary_reference_table, ["Параметр", "Значение"], reference_rows)
        self.summary_analytic_label.setText(report.analytic_note or "—")
        self.summary_reference_box.setVisible(
            report.reference_x is not None or report.reference_f is not None or bool(report.analytic_note)
        )

        skipped_rows = [
            [
                METHOD_SPECS[item.method_key].title,
                f"{item.eps:g}",
                f"{item.l:g}",
                item.reason,
            ]
            for item in report.skipped_runs
        ]
        self._set_table_data(self.summary_skipped_table, ["Метод", "ε", "l", "Причина"], skipped_rows)
        self.summary_skipped_box.setVisible(bool(skipped_rows))

        note_items = list(report.observations)
        self.summary_notes_list.clear()
        for note in note_items:
            self.summary_notes_list.addItem(note)
        self.summary_notes_box.setVisible(bool(note_items))

    def _set_defaults(self) -> None:
        self.function_buttons[0].setChecked(True)
        self.kind_buttons[0].setChecked(True)
        self.method_buttons[0].setChecked(True)
        self.a_input.setText("-10")
        self.b_input.setText("10")
        self.eps_input.setText("0.01")
        self.l_input.setText("0.1")
        self._populate_summary()

    def _selected_value(self, buttons: List[QRadioButton], default: str) -> str:
        for button in buttons:
            if button.isChecked():
                return str(button.property("choice_value"))
        return default

    def _collect_input_config(self) -> InputConfig:
        config = build_input_config(
            function_key=self._selected_value(self.function_buttons, "F1"),
            kind=self._selected_value(self.kind_buttons, "max"),
            method_key=self._selected_value(self.method_buttons, "all"),
            a_raw=self.a_input.text(),
            b_raw=self.b_input.text(),
            eps_raw=self.eps_input.text(),
            l_raw=self.l_input.text(),
        )
        return config

    def _set_busy(self, busy: bool, text: str) -> None:
        logger.info("Set busy=%s text=%s", busy, text)
        self.state.busy = busy
        for widget in self.control_widgets:
            widget.setEnabled(not busy)
        if not busy:
            self.plot_button.setEnabled(self.state.last_report is not None)
        self.status_label.setText(text)

    def on_function_change(self) -> None:
        if self._selected_value(self.function_buttons, "F1") == "F1":
            self.kind_buttons[0].setChecked(True)

    def _start_worker(
        self,
        label: str,
        task: Callable[[], object],
        on_success: Callable[[object], None],
        worker_type: str,
    ) -> None:
        if self.state.busy and worker_type == "calc":
            return

        thread = QThread(self)
        worker = Worker(task, label)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.failed.connect(self._handle_worker_error)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)

        if worker_type == "calc":
            self.calc_thread = thread
            self.calc_worker = worker
            self._set_busy(True, label)
        else:
            self.plot_thread = thread
            self.plot_worker = worker
            self.plot_state_label.setText("Строю график...")

        thread.start()

    def _handle_worker_error(self, message: str, stack: str) -> None:
        logger.error("Worker error message=%s", message)
        logger.debug("Worker traceback\n%s", stack)
        self._set_busy(False, "Ошибка")
        self.plot_state_label.setText("Ошибка построения графика.")
        QMessageBox.critical(self, "Ошибка", message)

    def handle_run_single(self) -> None:
        logger.info("Single run requested")
        try:
            config = self._collect_input_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self._start_worker("Выполняется расчёт...", lambda: run_single(config), self._apply_report, "calc")

    def handle_run_full_grid(self) -> None:
        logger.info("Grid run requested")
        try:
            config = self._collect_input_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self._start_worker("Запускаю исследование ε × l...", lambda: run_full_grid(config), self._apply_report, "calc")

    def _apply_report(self, payload: object) -> None:
        report = payload
        if not isinstance(report, RunReport):
            return

        logger.info("Applying report methods=%s", report.method_keys)
        self.state.last_report = report
        self.state.selected_table_method = report.default_method_key or ""
        self.state.selected_grid_run_index = 0
        self.current_plot_version += 1
        self._populate_summary()
        self._rebuild_method_buttons(report)
        self._rebuild_grid_run_list()
        self._populate_iterations()
        self._show_plot_placeholder("График готов к построению. Открой вкладку или нажми 'Обновить график'.")
        self._set_busy(False, "Расчёт завершён")
        if self.tabs.currentIndex() == 2:
            self.render_last_plot()

    def _selected_grid_run(self) -> Optional[GridRunResult]:
        report = self.state.last_report
        if report is None or report.mode != "grid":
            return None
        runs = report.grid_runs_by_method.get(self.state.selected_table_method, ())
        if not runs:
            return None
        index = min(max(self.state.selected_grid_run_index, 0), len(runs) - 1)
        self.state.selected_grid_run_index = index
        return runs[index]

    def _rebuild_method_buttons(self, report: RunReport) -> None:
        self._clear_layout(self.table_method_buttons_layout)
        self.table_method_buttons = []

        for method_key in report.method_keys:
            button = QRadioButton(METHOD_SPECS[method_key].title)
            button.setProperty("choice_value", method_key)
            button.toggled.connect(self.on_table_method_change)
            self.table_method_buttons_layout.addWidget(button)
            self.table_method_buttons.append(button)
            if method_key == self.state.selected_table_method:
                button.setChecked(True)

        self.table_method_buttons_layout.addStretch(1)

    def _rebuild_grid_run_list(self) -> None:
        self.grid_run_list.blockSignals(True)
        self.grid_run_list.clear()
        report = self.state.last_report
        if report is None or report.mode != "grid":
            self.grid_run_list.hide()
            self.grid_run_list.blockSignals(False)
            return

        runs = report.grid_runs_by_method.get(self.state.selected_table_method, ())
        for run in runs:
            self.grid_run_list.addItem(
                f"ε={run.eps:g}, l={run.l:g} | вызовов={run.result.func_evals} | x*={run.result.x_opt:.6f}"
            )
        self.grid_run_list.setVisible(bool(runs))
        if runs:
            index = min(self.state.selected_grid_run_index, len(runs) - 1)
            self.grid_run_list.setCurrentRow(index)
            self.state.selected_grid_run_index = index
        self.grid_run_list.blockSignals(False)

    def _clear_layout(self, layout: QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)  # type: ignore[arg-type]

    def on_table_method_change(self) -> None:
        report = self.state.last_report
        if report is None:
            return
        for button in self.table_method_buttons:
            if button.isChecked():
                self.state.selected_table_method = str(button.property("choice_value"))
                break
        self.state.selected_grid_run_index = 0
        self._rebuild_grid_run_list()
        self._populate_iterations()

    def on_grid_run_change(self, row: int) -> None:
        if row < 0:
            return
        self.state.selected_grid_run_index = row
        self._populate_iterations()
        if self.tabs.currentIndex() == 2:
            self.render_last_plot()

    def _populate_iterations(self) -> None:
        self.iterations_tree.clear()
        report = self.state.last_report
        if report is None:
            return
        if report.mode == "grid":
            run = self._selected_grid_run()
            result = run.result if run is not None else None
        else:
            result = report.results_by_method.get(self.state.selected_table_method)
        if result is None:
            return
        for row in result.iterations:
            QTreeWidgetItem(
                self.iterations_tree,
                [
                    str(row.k),
                    f"{row.a:.8f}",
                    f"{row.b:.8f}",
                    f"{row.lam:.8f}",
                    f"{row.mu:.8f}",
                    f"{row.f_lam:.8f}",
                    f"{row.f_mu:.8f}",
                ],
            )
        for index in range(self.iterations_tree.columnCount()):
            self.iterations_tree.resizeColumnToContents(index)

    def _show_plot_placeholder(self, text: str) -> None:
        self.plot_state_label.setText(text)
        self.plot_state_label.show()
        if self.plot_canvas is not None:
            self.plot_host_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()
            self.plot_canvas = None

    def render_last_plot(self) -> None:
        report = self.state.last_report
        if report is None:
            QMessageBox.information(self, "График", "Сначала выполните расчёт.")
            return
        if self.plot_thread is not None and self.plot_thread.isRunning():
            return

        version = self.current_plot_version

        def build() -> object:
            if report.mode == "grid":
                current_run = self._selected_grid_run()
                plot_results = [current_run.result] if current_run is not None else []
            else:
                plot_results = list(report.results_by_method.values())
            return version, build_plot_figure(
                function_spec=report.function_spec,
                results=plot_results,
                plot_range=report.plot_range,
            )

        self._start_worker("Строю график...", build, self._finish_plot_render, "plot")

    def _finish_plot_render(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        version, figure = payload
        if version != self.current_plot_version:
            return
        if not isinstance(figure, Figure):
            return
        self._show_plot_placeholder("")
        self.plot_state_label.hide()
        self.plot_canvas = PlotCanvas(figure, self.plot_host)
        self.plot_host_layout.addWidget(self.plot_canvas)
        self.plot_canvas.draw()
        self.plot_button.setEnabled(True)

    def on_tab_changed(self, index: int) -> None:
        if index == 1:
            self._populate_iterations()
        elif index == 2 and self.state.last_report is not None:
            self.render_last_plot()

    def clear_output(self) -> None:
        if self.state.busy:
            return
        logger.info("Clearing window state")
        self.state.last_report = None
        self.state.selected_table_method = ""
        self.state.selected_grid_run_index = 0
        self._populate_summary()
        self.iterations_tree.clear()
        self.grid_run_list.clear()
        self.grid_run_list.hide()
        self._show_plot_placeholder("График появится после расчёта.")
        self._clear_layout(self.table_method_buttons_layout)
        self.table_method_buttons = []
        self.plot_button.setEnabled(False)
        self.status_label.setText("Очищено")


def main() -> None:
    logger.info("Qt GUI main start")
    app = QApplication.instance() or QApplication(sys.argv)
    window = ExtremumWindow()
    window.show()
    app.exec()
