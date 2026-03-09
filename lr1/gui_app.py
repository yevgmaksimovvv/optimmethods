import logging
import sys
import traceback
from typing import Callable, Dict, List, Optional

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
    QStackedWidget,
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
    from .function_defs import FUNCTION_TEMPLATE_SPECS
    from .logging_setup import configure_logging, get_log_file_path
    from .plotting import build_plot_figure
    from .search_methods import METHOD_SPECS
else:
    from app_models import AppState, GridRunResult, InputConfig, RunReport
    from app_service import build_input_config, run_full_grid, run_single
    from function_defs import FUNCTION_TEMPLATE_SPECS
    from logging_setup import configure_logging, get_log_file_path
    from plotting import build_plot_figure
    from search_methods import METHOD_SPECS


configure_logging()
logger = logging.getLogger("lr1.gui")


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
        self.coefficient_inputs: Dict[str, Dict[str, QLineEdit]] = {}
        self.function_stack_indexes: Dict[str, int] = {}

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
                font-family: "Helvetica Neue", "Arial", sans-serif;
                font-size: 15px;
            }
            QLabel {
                background: transparent;
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
                padding: 9px 14px;
                color: #f5f7fb;
                font-weight: 600;
                min-height: 22px;
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
            QPushButton[variant="primary"] {
                background: #1f6feb;
                border-color: #2f81f7;
            }
            QPushButton[variant="primary"]:hover {
                background: #2f81f7;
            }
            QPushButton[variant="primary"]:pressed {
                background: #1858ba;
            }
            QPushButton[role="action"] {
                padding: 8px 16px;
                min-height: 22px;
                font-size: 15px;
            }
            QPushButton[role="chip"] {
                background: #414b67;
                border-color: #627095;
                border-radius: 10px;
                padding: 4px 10px;
                min-height: 0;
                min-width: 0;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton[role="chip"]:hover {
                background: #4d5a7b;
            }
            QPushButton[role="chip"]:pressed {
                background: #39435b;
            }
            QLineEdit, QListWidget {
                background: #14161b;
                border: 1px solid #464b59;
                border-radius: 8px;
                padding: 7px 10px;
                color: #f5f7fb;
                selection-background-color: #2a6df4;
            }
            QTreeWidget, QTableWidget {
                background: #14161b;
                border: 1px solid #464b59;
                border-radius: 8px;
                padding: 0;
                color: #f5f7fb;
                gridline-color: #2d3241;
                selection-background-color: #2a6df4;
                alternate-background-color: #1b1f29;
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
                border: 0;
                top: 0;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar {
                qproperty-drawBase: 0;
                qproperty-expanding: 1;
            }
            QTabBar::tab {
                background: #2c303b;
                border: 1px solid #4b4f5c;
                padding: 8px 16px 10px 16px;
                margin-right: 6px;
                margin-bottom: 2px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                min-width: 0px;
                min-height: 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #46516b;
                border-color: #647596;
                color: #ffffff;
            }
            QLabel#SectionHint {
                color: #c8cfdb;
                font-size: 14px;
                padding: 4px 2px;
            }
            QLabel#SectionCaption {
                color: #9aa5bb;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            QLabel#SummaryEmptyLabel {
                background: #171a22;
                border: 1px dashed #434a5c;
                border-radius: 12px;
                color: #c8cfdb;
                font-size: 15px;
            }
            QLabel#SummaryEmptyTitle {
                color: #eef2f8;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#SummaryEmptyText {
                color: #b8c1d1;
                font-size: 15px;
            }
            QWidget#SummaryEmptyCard {
                background: #181b24;
                border: 1px solid #31384a;
                border-radius: 14px;
            }
            QWidget#FunctionFormulaCard, QWidget#CoefficientCard {
                background: #181b24;
                border: 1px solid #31384a;
                border-radius: 12px;
            }
            QWidget#CoeffCell {
                background: transparent;
            }
            QLabel#FunctionFormulaLabel {
                color: #eef2f8;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#CoeffCaption {
                color: #c8cfdb;
                font-size: 13px;
                font-weight: 700;
            }
            QLineEdit[role="coefficient"] {
                font-size: 15px;
                font-weight: 600;
                padding: 8px 10px;
            }
            QRadioButton {
                spacing: 7px;
                padding: 2px 0;
                min-height: 22px;
                font-size: 15px;
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
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_panel.setMinimumWidth(500)
        left_panel.setMaximumWidth(560)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self._build_function_box(left_layout)
        self._build_scenario_box(left_layout)
        self._build_parameters_box(left_layout)
        self._build_actions_box(left_layout)

        left_layout.addStretch(1)
        left_scroll.setWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self._build_tabs(right_layout)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([540, 960])

    def _build_function_box(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Функция")
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(16, 16, 16, 14)
        box_layout.setSpacing(8)

        selector_holder = QWidget()
        selector_layout = QHBoxLayout(selector_holder)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(16)

        self.function_group = QButtonGroup(self)
        self.function_buttons = []
        for caption, value in (
            ("Квадратичная", "quadratic"),
            ("Рациональная", "rational"),
        ):
            button = QRadioButton(caption)
            button.setProperty("choice_value", value)
            button.toggled.connect(self.on_function_change)
            self.function_group.addButton(button)
            selector_layout.addWidget(button)
            self.function_buttons.append(button)
        selector_layout.addStretch(1)
        box_layout.addWidget(selector_holder)

        self._build_function_editors(box_layout)
        layout.addWidget(box)

    def _build_scenario_box(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Сценарий")
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        grid = QGridLayout(box)
        grid.setContentsMargins(16, 16, 16, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 100)
        grid.setColumnStretch(1, 1)

        self.kind_group = QButtonGroup(self)
        self.kind_buttons = self._create_radio_grid(
            grid,
            row=0,
            label="Цель",
            group=self.kind_group,
            options=(("Максимум", "max"), ("Минимум", "min")),
            columns=2,
        )
        self.method_group = QButtonGroup(self)
        self.method_buttons = self._create_radio_grid(
            grid,
            row=1,
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
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        grid = QGridLayout(box)
        grid.setContentsMargins(16, 16, 16, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 150)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 1)

        self.a_input, self.b_input = self._add_interval_inputs(grid, 0, "Границы")
        self.eps_input = self._add_labeled_input(grid, 1, "Точность ε")
        self.l_input = self._add_labeled_input(grid, 2, "Длина интервала L")

        layout.addWidget(box)

    def _build_actions_box(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Запуск")
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(16, 16, 16, 16)
        box_layout.setSpacing(10)

        self.run_button = QPushButton("Рассчитать")
        self.run_button.clicked.connect(self.handle_run_single)
        self.run_button.setProperty("variant", "primary")
        self.run_button.setProperty("role", "action")
        self.run_button.setMinimumHeight(38)
        box_layout.addWidget(self.run_button)

        self.grid_button = QPushButton("Серия расчётов")
        self.grid_button.clicked.connect(self.handle_run_full_grid)
        self.grid_button.setProperty("role", "action")
        self.grid_button.setMinimumHeight(38)
        box_layout.addWidget(self.grid_button)

        self.actions_hint_label = QLabel(
            "Серия расчётов автоматически прогоняет несколько сочетаний ε и l\n"
            "и показывает, как меняется результат у разных методов."
        )
        self.actions_hint_label.setObjectName("SectionHint")
        self.actions_hint_label.setWordWrap(True)
        box_layout.addWidget(self.actions_hint_label)

        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self.clear_output)
        self.clear_button.setProperty("role", "action")
        self.clear_button.setMinimumHeight(38)
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
                self.actions_hint_label,
                self.clear_button,
                *self.function_buttons,
                *self.kind_buttons,
                *self.method_buttons,
            ]
        )

    def _build_tabs(self, layout: QVBoxLayout) -> None:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setUsesScrollButtons(False)
        layout.addWidget(self.tabs)

        summary_tab = QWidget()
        summary_tab_layout = QVBoxLayout(summary_tab)
        summary_tab_layout.setContentsMargins(0, 0, 0, 0)
        summary_tab_layout.setSpacing(0)
        self.summary_stack = QStackedWidget()
        summary_tab_layout.addWidget(self.summary_stack)

        summary_empty_page = QWidget()
        summary_empty_layout = QVBoxLayout(summary_empty_page)
        summary_empty_layout.setContentsMargins(16, 24, 16, 0)
        summary_empty_layout.setSpacing(0)

        empty_card = QWidget()
        empty_card.setObjectName("SummaryEmptyCard")
        empty_card.setMaximumWidth(700)
        empty_card.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        empty_card_layout = QVBoxLayout(empty_card)
        empty_card_layout.setContentsMargins(30, 24, 30, 24)
        empty_card_layout.setSpacing(18)

        empty_title = QLabel("Пока нет результатов")
        empty_title.setObjectName("SummaryEmptyTitle")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        empty_card_layout.addWidget(empty_title)

        empty_text = QLabel(
            "Слева выбери функцию, диапазон и метод.\n"
            "После запуска здесь появятся результаты и сравнение методов."
        )
        empty_text.setObjectName("SummaryEmptyText")
        empty_text.setAlignment(Qt.AlignCenter)
        empty_text.setWordWrap(True)
        empty_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        empty_card_layout.addWidget(empty_text)

        self.summary_empty_label = QLabel(
            "Рассчитать для текущих параметров или запусти серию расчётов."
        )
        self.summary_empty_label.setObjectName("SectionHint")
        self.summary_empty_label.setAlignment(Qt.AlignCenter)
        self.summary_empty_label.setWordWrap(True)
        self.summary_empty_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.summary_empty_label.setContentsMargins(6, 0, 6, 0)
        empty_card_layout.addWidget(self.summary_empty_label)

        empty_card_row = QHBoxLayout()
        empty_card_row.setContentsMargins(0, 0, 0, 0)
        empty_card_row.setSpacing(0)
        empty_card_row.addStretch(1)
        empty_card_row.addWidget(empty_card, 1)
        empty_card_row.addStretch(1)
        summary_empty_layout.addLayout(empty_card_row)
        summary_empty_layout.addStretch(1)
        self.summary_stack.addWidget(summary_empty_page)

        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setFrameShape(QScrollArea.NoFrame)
        summary_content = QWidget()
        summary_layout = QVBoxLayout(summary_content)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(12)

        self.summary_results_box = QGroupBox("Результаты")
        self.summary_results_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        results_layout = QVBoxLayout(self.summary_results_box)
        self.summary_results_table = self._create_table_widget()
        self.summary_results_table.setProperty("max_visible_rows", 6)
        results_layout.addWidget(self.summary_results_table)
        summary_layout.addWidget(self.summary_results_box)

        self.summary_reference_box = QGroupBox("Теоретический ориентир")
        self.summary_reference_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        reference_layout = QVBoxLayout(self.summary_reference_box)
        self.summary_reference_table = self._create_table_widget()
        self.summary_reference_table.setProperty("max_visible_rows", 4)
        self.summary_analytic_label = QLabel("—")
        self.summary_analytic_label.setObjectName("SectionHint")
        self.summary_analytic_label.setWordWrap(True)
        reference_layout.addWidget(self.summary_reference_table)
        reference_layout.addWidget(self.summary_analytic_label)
        summary_layout.addWidget(self.summary_reference_box)

        self.summary_skipped_box = QGroupBox("Пропущенные наборы параметров")
        self.summary_skipped_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        skipped_layout = QVBoxLayout(self.summary_skipped_box)
        self.summary_skipped_table = self._create_table_widget()
        self.summary_skipped_table.setProperty("max_visible_rows", 5)
        skipped_layout.addWidget(self.summary_skipped_table)
        summary_layout.addWidget(self.summary_skipped_box)

        self.summary_notes_box = QGroupBox("Выводы")
        self.summary_notes_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        notes_layout = QVBoxLayout(self.summary_notes_box)
        self.summary_notes_list = QListWidget()
        self.summary_notes_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.summary_notes_list.setFocusPolicy(Qt.NoFocus)
        self.summary_notes_list.setProperty("max_visible_rows", 5)
        notes_layout.addWidget(self.summary_notes_list)
        summary_layout.addWidget(self.summary_notes_box)
        summary_layout.addStretch(1)

        summary_scroll.setWidget(summary_content)
        self.summary_stack.addWidget(summary_scroll)
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
        self.iterations_tree.setHeaderLabels(("k", "a(k)", "b(k)", "λ(k)", "μ(k)", "F(λ(k))", "F(μ(k))"))
        self.iterations_tree.setAlternatingRowColors(True)
        self.iterations_tree.setRootIsDecorated(False)
        self.iterations_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.iterations_tree.setProperty("max_visible_rows", 14)
        self.grid_run_list = QListWidget()
        self.grid_run_list.setProperty("max_visible_rows", 5)
        self.grid_run_list.hide()
        self.grid_run_list.currentRowChanged.connect(self.on_grid_run_change)
        iterations_layout.addWidget(self.grid_run_list)
        iterations_layout.addWidget(self.iterations_tree)
        iterations_layout.addStretch(1)
        self.tabs.addTab(iterations_tab, "Итерации")

        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.plot_state_label = QLabel("График появится после расчёта.")
        self.plot_state_label.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.plot_state_label)
        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setFrameShape(QScrollArea.NoFrame)
        plot_layout.addWidget(self.plot_scroll)
        self.plot_host = QWidget()
        self.plot_host_layout = QVBoxLayout(self.plot_host)
        self.plot_host_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_host_layout.setSpacing(0)
        self.plot_scroll.setWidget(self.plot_host)
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
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(112)
        label_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(label_widget, row, 0)
        holder = QWidget()
        holder_layout = QHBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(14)
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
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(112)
        label_widget.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        grid.addWidget(label_widget, row, 0, Qt.AlignTop)
        holder = QWidget()
        option_width = 250
        holder_layout = QGridLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setHorizontalSpacing(28)
        holder_layout.setVerticalSpacing(8)
        holder_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        buttons: List[QRadioButton] = []
        for index, (caption, value) in enumerate(options):
            button = QRadioButton(caption)
            button.setProperty("choice_value", value)
            group.addButton(button)
            button.setFixedWidth(option_width)
            holder_layout.addWidget(button, index // columns, index % columns)
            buttons.append(button)
        for column in range(columns):
            holder_layout.setColumnMinimumWidth(column, option_width)
        grid.addWidget(holder, row, 1, 1, 3)
        return buttons

    def _build_function_editors(self, layout: QVBoxLayout) -> None:
        self.function_editor_stack = QStackedWidget()
        self.function_editor_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout.addWidget(self.function_editor_stack)

        for index, template in enumerate(FUNCTION_TEMPLATE_SPECS.values()):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(8)

            formula_card = QWidget()
            formula_card.setObjectName("FunctionFormulaCard")
            formula_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            formula_layout = QVBoxLayout(formula_card)
            formula_layout.setContentsMargins(14, 10, 14, 10)
            formula_layout.setSpacing(8)

            formula_caption = QLabel("Формула")
            formula_caption.setObjectName("SectionCaption")
            formula_layout.addWidget(formula_caption)

            hint = QLabel(template.formula_hint)
            hint.setObjectName("FunctionFormulaLabel")
            hint.setAlignment(Qt.AlignCenter)
            hint.setWordWrap(True)
            hint.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            formula_layout.addWidget(hint)
            page_layout.addWidget(formula_card)

            coeff_card = QWidget()
            coeff_card.setObjectName("CoefficientCard")
            coeff_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            coeff_card_layout = QVBoxLayout(coeff_card)
            coeff_card_layout.setContentsMargins(14, 10, 14, 12)
            coeff_card_layout.setSpacing(6)

            coeff_caption = QLabel("Коэффициенты")
            coeff_caption.setObjectName("SectionCaption")
            coeff_card_layout.addWidget(coeff_caption)

            coeff_grid = QGridLayout()
            coeff_grid.setContentsMargins(0, 0, 0, 0)
            coeff_grid.setHorizontalSpacing(10)
            coeff_grid.setVerticalSpacing(10)
            coeff_grid.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.coefficient_inputs[template.key] = {}

            for coeff_index, coefficient in enumerate(template.coefficients):
                col = coeff_index % 3
                row_base = coeff_index // 3
                cell = QWidget()
                cell.setObjectName("CoeffCell")
                cell.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(4)

                coeff_label = QLabel(coefficient.label)
                coeff_label.setObjectName("CoeffCaption")
                coeff_label.setAlignment(Qt.AlignCenter)
                coeff_input = QLineEdit()
                coeff_input.setProperty("role", "coefficient")
                coeff_input.setMinimumWidth(88)
                coeff_input.setMaximumWidth(108)
                coeff_input.setMinimumHeight(36)
                coeff_input.setAlignment(Qt.AlignCenter)
                coeff_input.setText(f"{coefficient.default:g}")
                cell_layout.addWidget(coeff_label)
                cell_layout.addWidget(coeff_input)
                coeff_grid.addWidget(cell, row_base, col)
                self.coefficient_inputs[template.key][coefficient.key] = coeff_input
                self.control_widgets.append(coeff_input)

            coeff_row = QHBoxLayout()
            coeff_row.setContentsMargins(0, 0, 0, 0)
            coeff_row.setSpacing(0)
            coeff_row.addStretch(1)
            coeff_row.addLayout(coeff_grid)
            coeff_row.addStretch(1)
            coeff_card_layout.addLayout(coeff_row)
            page_layout.addWidget(coeff_card)
            self.function_editor_stack.addWidget(page)
            self.function_stack_indexes[template.key] = index

    def _add_interval_inputs(self, grid: QGridLayout, row: int, label: str) -> tuple[QLineEdit, QLineEdit]:
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(150)
        label_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(label_widget, row, 0)

        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        left_caption = QLabel("a")
        left_caption.setObjectName("CoeffCaption")
        left_caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(left_caption)

        left_line = QLineEdit()
        left_line.setMaximumWidth(120)
        left_line.setMinimumHeight(38)
        layout.addWidget(left_line)

        right_caption = QLabel("b")
        right_caption.setObjectName("CoeffCaption")
        right_caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(right_caption)

        right_line = QLineEdit()
        right_line.setMaximumWidth(120)
        right_line.setMinimumHeight(38)
        layout.addWidget(right_line)
        layout.addStretch(1)

        grid.addWidget(holder, row, 1, 1, 2)
        return left_line, right_line

    def _add_labeled_input(self, grid: QGridLayout, row: int, label: str) -> QLineEdit:
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(150)
        label_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label_widget.setWordWrap(True)
        grid.addWidget(label_widget, row, 0)

        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        line = QLineEdit()
        line.setFixedWidth(180)
        line.setMinimumHeight(38)
        layout.addWidget(line)
        layout.addStretch(1)

        grid.addWidget(holder, row, 1, 1, 2)
        return line

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
        self._fit_table_height(table)

    def _fit_table_height(self, table: QTableWidget) -> None:
        max_visible_rows = int(table.property("max_visible_rows") or 6)
        visible_rows = min(table.rowCount(), max_visible_rows)
        header_height = table.horizontalHeader().height() if table.columnCount() else 0
        rows_height = sum(table.rowHeight(index) for index in range(visible_rows))
        frame_height = table.frameWidth() * 2
        base_padding = 10

        if table.rowCount() == 0:
            empty_height = max(header_height + frame_height + 72, 120)
            table.setFixedHeight(empty_height)
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            return

        height = header_height + rows_height + frame_height + base_padding
        table.setFixedHeight(min(max(height, 96), 320))
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if table.rowCount() > max_visible_rows else Qt.ScrollBarAlwaysOff
        )

    def _format_float(self, value: float, digits: int = 6) -> str:
        return f"{value:.{digits}f}"

    def _format_interval(self, interval) -> str:
        return f"[{self._format_float(interval[0], 5)}, {self._format_float(interval[1], 5)}]"

    def _format_optional_float(self, value: Optional[float]) -> str:
        return "—" if value is None else self._format_float(value)

    def _result_error_pair(self, report: RunReport, x_value: float, f_value: float) -> List[str]:
        if report.reference_x is None or report.reference_f is None:
            return ["—", "—"]
        return [
            self._format_float(abs(x_value - report.reference_x)),
            self._format_float(abs(f_value - report.reference_f)),
        ]

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
                    self._format_float(result.x_opt),
                    self._format_float(result.f_opt),
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
                        self._format_float(run.result.x_opt),
                        self._format_float(run.result.f_opt),
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
            self.summary_stack.setCurrentIndex(0)
            self.summary_results_box.hide()
            self._set_table_data(self.summary_results_table, ["Результат"], [])
            self._set_table_data(self.summary_reference_table, ["Параметр", "Значение"], [])
            self._set_table_data(self.summary_skipped_table, ["Метод", "ε", "l", "Причина"], [])
            self.summary_reference_box.hide()
            self.summary_skipped_box.hide()
            self.summary_notes_box.hide()
            self.summary_analytic_label.setText("—")
            self.summary_notes_list.clear()
            return

        self.summary_stack.setCurrentIndex(1)
        self.summary_results_box.show()
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
                "|Δx|",
                "|Δf|",
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
                "|Δx|",
                "|Δf|",
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
        self._fit_list_height(self.summary_notes_list)
        self.summary_notes_box.setVisible(bool(note_items))

    def _fit_list_height(self, widget: QListWidget) -> None:
        max_visible_rows = int(widget.property("max_visible_rows") or 5)
        if widget.count() == 0:
            widget.setFixedHeight(80)
            return
        row_height = widget.sizeHintForRow(0)
        if row_height <= 0:
            row_height = 28
        visible_rows = min(widget.count(), max_visible_rows)
        frame_height = widget.frameWidth() * 2
        height = (row_height * visible_rows) + frame_height + 12
        widget.setFixedHeight(min(max(height, 72), 240))

    def _fit_tree_height(self, tree: QTreeWidget) -> None:
        max_visible_rows = int(tree.property("max_visible_rows") or 14)
        visible_rows = min(tree.topLevelItemCount(), max_visible_rows)
        header_height = tree.header().height() if tree.columnCount() else 0
        rows_height = 0
        for index in range(visible_rows):
            rows_height += tree.sizeHintForIndex(tree.indexFromItem(tree.topLevelItem(index), 0)).height()
        frame_height = tree.frameWidth() * 2

        if tree.topLevelItemCount() == 0:
            tree.setFixedHeight(max(header_height + frame_height + 96, 140))
            tree.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            return

        height = header_height + rows_height + frame_height + 12
        tree.setFixedHeight(min(max(height, 140), 520))
        tree.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if tree.topLevelItemCount() > max_visible_rows else Qt.ScrollBarAlwaysOff
        )

    def _reset_function_defaults(self) -> None:
        for template in FUNCTION_TEMPLATE_SPECS.values():
            editors = self.coefficient_inputs.get(template.key, {})
            for coefficient in template.coefficients:
                editor = editors.get(coefficient.key)
                if editor is not None:
                    editor.setText(f"{coefficient.default:g}")

    def _set_defaults(self) -> None:
        self._reset_function_defaults()
        self.function_buttons[0].setChecked(True)
        self.kind_buttons[0].setChecked(True)
        self.method_buttons[0].setChecked(True)
        self.a_input.setText("-10")
        self.b_input.setText("10")
        self.eps_input.setText("0.01")
        self.l_input.setText("0.1")
        self.on_function_change()
        self._populate_summary()

    def _selected_value(self, buttons: List[QRadioButton], default: str) -> str:
        for button in buttons:
            if button.isChecked():
                return str(button.property("choice_value"))
        return default

    def _selected_function_key(self) -> str:
        return self._selected_value(self.function_buttons, "quadratic")

    def _collect_coefficient_raws(self, function_key: str) -> Dict[str, str]:
        return {
            key: editor.text()
            for key, editor in self.coefficient_inputs.get(function_key, {}).items()
        }

    def _collect_input_config(self) -> InputConfig:
        function_key = self._selected_function_key()
        config = build_input_config(
            function_key=function_key,
            kind=self._selected_value(self.kind_buttons, "max"),
            method_key=self._selected_value(self.method_buttons, "all"),
            a_raw=self.a_input.text(),
            b_raw=self.b_input.text(),
            eps_raw=self.eps_input.text(),
            l_raw=self.l_input.text(),
            coefficient_raws=self._collect_coefficient_raws(function_key),
        )
        return config

    def _set_busy(self, busy: bool, text: str) -> None:
        logger.info("Set busy=%s text=%s", busy, text)
        self.state.busy = busy
        for widget in self.control_widgets:
            widget.setEnabled(not busy)

    def on_function_change(self) -> None:
        function_key = self._selected_function_key()
        index = self.function_stack_indexes.get(function_key)
        if index is not None:
            self.function_editor_stack.setCurrentIndex(index)
            self._sync_function_editor_height()

    def _sync_function_editor_height(self) -> None:
        current_page = self.function_editor_stack.currentWidget()
        if current_page is None:
            return
        self.function_editor_stack.setFixedHeight(current_page.sizeHint().height())

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
            thread.finished.connect(lambda: self._release_worker_refs("calc", thread, worker))
            self._set_busy(True, label)
        else:
            self.plot_thread = thread
            self.plot_worker = worker
            thread.finished.connect(lambda: self._release_worker_refs("plot", thread, worker))
            self.plot_state_label.setText("Строю график...")

        thread.start()

    def _release_worker_refs(self, worker_type: str, thread: QThread, worker: Worker) -> None:
        if worker_type == "calc":
            if self.calc_thread is thread:
                self.calc_thread = None
            if self.calc_worker is worker:
                self.calc_worker = None
            return
        if self.plot_thread is thread:
            self.plot_thread = None
        if self.plot_worker is worker:
            self.plot_worker = None

    def _thread_is_running(self, thread: Optional[QThread]) -> bool:
        if thread is None:
            return False
        try:
            return thread.isRunning()
        except RuntimeError:
            return False

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
        self._show_plot_placeholder("Открой вкладку «График» для автоматического построения.")
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
            self._fit_list_height(self.grid_run_list)
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
        self._fit_list_height(self.grid_run_list)
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
        if self.tabs.currentIndex() == 2 and report.mode == "grid":
            self.render_last_plot()

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
            self._fit_tree_height(self.iterations_tree)
            return
        if report.mode == "grid":
            run = self._selected_grid_run()
            result = run.result if run is not None else None
        else:
            result = report.results_by_method.get(self.state.selected_table_method)
        if result is None:
            self._fit_tree_height(self.iterations_tree)
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
        self._fit_tree_height(self.iterations_tree)

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
        if self._thread_is_running(self.plot_thread):
            return

        version = self.current_plot_version

        def build() -> object:
            if report.mode == "grid":
                current_run = self._selected_grid_run()
                plot_results = [current_run.result] if current_run is not None else []
            else:
                plot_results = [
                    report.results_by_method[method_key]
                    for method_key in report.method_keys
                    if method_key in report.results_by_method
                ]
            return version, build_plot_figure(
                function_spec=report.function_spec,
                results=plot_results,
                plot_range=report.plot_range,
                reference_x=report.reference_x,
                reference_f=report.reference_f,
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
        width_px, height_px = figure.get_size_inches() * figure.dpi
        self.plot_canvas.setMinimumSize(int(width_px), int(height_px))
        self.plot_host_layout.addWidget(self.plot_canvas)
        self.plot_canvas.draw()

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


def main() -> None:
    logger.info("Qt GUI main start")
    app = QApplication.instance() or QApplication(sys.argv)
    window = ExtremumWindow()
    window.show()
    app.exec()
