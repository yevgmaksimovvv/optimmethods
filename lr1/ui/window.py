"""Главное окно настольного приложения.

Класс `ExtremumWindow` координирует весь пользовательский сценарий:
- собирает параметры задачи;
- запускает расчёт или серию расчётов в фоне;
- раскладывает `RunReport` по вкладкам;
- инициирует построение графиков при необходимости.

Само окно старается не выполнять математическую работу напрямую и делегирует
расчёты в прикладной и доменный слой.
"""

import logging
import sys
from typing import Dict, List, Optional

from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lr1.application.services import build_input_config, run_full_grid, run_single
from lr1.domain.functions import FUNCTION_TEMPLATE_SPECS
from lr1.domain.models import AppState, GridRunResult, InputConfig, RunReport
from lr1.domain.search import METHOD_SPECS
from lr1.infrastructure.logging import configure_logging, get_log_file_path
from lr1.infrastructure.settings import DEFAULT_INPUT_EPS, DEFAULT_INTERVAL, DEFAULT_L
from lr1.ui.plotting import build_grid_plot_figure, build_plot_figure
from lr1.ui.tabs import IterationsTab, PlotTab, SummaryTab
from lr1.ui.workers import TaskController


configure_logging()
logger = logging.getLogger("lr1.gui")

APP_TITLE = "Исследование методов поиска экстремумов"


class ExtremumWindow(QMainWindow):
    """Основное окно лабораторного приложения."""
    def __init__(self):
        """Создаёт окно, фоновые контроллеры и базовое состояние интерфейса."""
        super().__init__()
        logger.info("ExtremumWindow init start")
        self.state = AppState()
        self.current_plot_version = 0
        self.control_widgets: List[QWidget] = []
        self.coefficient_inputs: Dict[str, Dict[str, QLineEdit]] = {}
        self.function_stack_indexes: Dict[str, int] = {}

        self.calc_task = TaskController(self)
        self.calc_task.succeeded.connect(self._apply_report)
        self.calc_task.failed.connect(self._handle_worker_error)
        self.plot_task = TaskController(self)
        self.plot_task.succeeded.connect(self._finish_plot_render)
        self.plot_task.failed.connect(self._handle_worker_error)

        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 940)
        self._apply_styles()
        self._build_ui()
        self._set_defaults()
        logger.info("ExtremumWindow init done log_file=%s", get_log_file_path())

    def _apply_styles(self) -> None:
        """Подключает единую таблицу стилей для всего интерфейса."""
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
            QLineEdit, QListWidget {
                background: #14161b;
                border: 1px solid #464b59;
                border-radius: 8px;
                padding: 7px 10px;
                color: #f5f7fb;
                selection-background-color: #2a6df4;
            }
            QListWidget#GridRunList {
                background: transparent;
                border: 0;
                padding: 0;
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
            QListWidget#GridRunList::item {
                padding: 0;
                margin: 0 0 10px 0;
                border: 0;
                background: transparent;
            }
            QWidget#GridRunCard {
                background: #171b24;
                border: 1px solid #31384a;
                border-radius: 12px;
            }
            QWidget#GridRunCard[selected="true"] {
                background: #1b3c73;
                border: 1px solid #2f81f7;
            }
            QWidget#GridRunMetric {
                background: transparent;
            }
            QLabel[role="grid-run-metric-caption"] {
                color: #9aa5bb;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            QLabel[role="grid-run-metric-value"] {
                color: #eef2f8;
                font-size: 17px;
                font-weight: 700;
                padding-top: 1px;
            }
            QWidget#GridRunCard[selected="true"] QLabel[role="grid-run-metric-caption"] {
                color: #cddcff;
            }
            QWidget#GridRunCard[selected="true"] QLabel[role="grid-run-metric-value"] {
                color: #ffffff;
            }
            QTreeWidget#IterationsTree {
                background: #12161d;
                alternate-background-color: #151b25;
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
                font-size: 14px;
            }
            QTableWidget[variant="report"] {
                background: #12161d;
                alternate-background-color: #151b25;
                font-size: 14px;
            }
            QTreeWidget#IterationsTree::item {
                padding: 5px 8px;
                border: 0;
            }
            QTableWidget[variant="report"]::item {
                padding: 5px 10px;
                border: 0;
            }
            QTreeWidget#IterationsTree::item:selected {
                background: #22314f;
                color: #ffffff;
            }
            QTreeWidget#IterationsTree QHeaderView::section {
                background: #222938;
                color: #dbe2ee;
                border-right: 1px solid #33415b;
                border-bottom: 1px solid #33415b;
                padding: 6px 8px;
                font-size: 12px;
                font-weight: 700;
            }
            QTableWidget[variant="report"] QHeaderView::section {
                background: #222938;
                color: #dbe2ee;
                border-right: 1px solid #33415b;
                border-bottom: 1px solid #33415b;
                padding: 6px 8px;
                font-size: 12px;
                font-weight: 700;
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
        """Собирает двухпанельный интерфейс: управление слева, результаты справа."""
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

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setUsesScrollButtons(False)
        right_layout.addWidget(self.tabs)

        self.summary_tab = SummaryTab()
        self.iterations_tab = IterationsTab(self.on_grid_run_change, self.on_table_method_change)
        self.plot_tab = PlotTab()
        self.tabs.addTab(self.summary_tab, "Сводка")
        self.tabs.addTab(self.iterations_tab, "Итерации")
        self.tabs.addTab(self.plot_tab, "График")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([540, 960])

    def _build_function_box(self, layout: QVBoxLayout) -> None:
        """Строит секцию выбора типа функции и редактирования коэффициентов."""
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
        for caption, value in (("Квадратичная", "quadratic"), ("Рациональная", "rational")):
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
        """Строит блок выбора цели поиска и алгоритма."""
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
        """Строит блок ручного ввода интервала, `ε` и `l`."""
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
        """Строит блок кнопок запуска, серии расчётов и очистки результата."""
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

    def _create_radio_grid(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        group: QButtonGroup,
        options,
        columns: int,
    ) -> List[QRadioButton]:
        """Создаёт сетку радиокнопок и возвращает список созданных элементов."""
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
        """Создаёт страницы редакторов коэффициентов для каждого типа функции."""
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
            coeff_row.addStretch(1)
            coeff_row.addLayout(coeff_grid)
            coeff_row.addStretch(1)
            coeff_card_layout.addLayout(coeff_row)
            page_layout.addWidget(coeff_card)
            self.function_editor_stack.addWidget(page)
            self.function_stack_indexes[template.key] = index

    def _add_interval_inputs(self, grid: QGridLayout, row: int, label: str) -> tuple[QLineEdit, QLineEdit]:
        """Добавляет в сетку пару полей для левой и правой границы интервала."""
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
        """Добавляет обычное подписанное поле ввода в таблицу параметров."""
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(150)
        label_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label_widget.setWordWrap(True)
        grid.addWidget(label_widget, row, 0)

        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)

        line = QLineEdit()
        line.setFixedWidth(180)
        line.setMinimumHeight(38)
        layout.addWidget(line)
        layout.addStretch(1)

        grid.addWidget(holder, row, 1, 1, 2)
        return line

    def _set_defaults(self) -> None:
        """Заполняет интерфейс начальными значениями по умолчанию."""
        self._reset_function_defaults()
        self.function_buttons[0].setChecked(True)
        self.kind_buttons[0].setChecked(True)
        self.method_buttons[0].setChecked(True)
        self.a_input.setText(DEFAULT_INTERVAL[0])
        self.b_input.setText(DEFAULT_INTERVAL[1])
        self.eps_input.setText(DEFAULT_INPUT_EPS)
        self.l_input.setText(DEFAULT_L)
        self.on_function_change()
        self.summary_tab.populate(None)

    def _reset_function_defaults(self) -> None:
        """Возвращает коэффициенты всех шаблонов функций к исходным значениям."""
        for template in FUNCTION_TEMPLATE_SPECS.values():
            editors = self.coefficient_inputs.get(template.key, {})
            for coefficient in template.coefficients:
                editor = editors.get(coefficient.key)
                if editor is not None:
                    editor.setText(f"{coefficient.default:g}")

    def _selected_value(self, buttons: List[QRadioButton], default: str) -> str:
        """Возвращает `choice_value` выбранной радиокнопки или значение по умолчанию."""
        for button in buttons:
            if button.isChecked():
                return str(button.property("choice_value"))
        return default

    def _selected_function_key(self) -> str:
        """Возвращает ключ текущего выбранного типа функции."""
        return self._selected_value(self.function_buttons, "quadratic")

    def _collect_coefficient_raws(self, function_key: str) -> Dict[str, str]:
        """Собирает сырые строки коэффициентов из активного редактора."""
        return {key: editor.text() for key, editor in self.coefficient_inputs.get(function_key, {}).items()}

    def _collect_input_config(self) -> InputConfig:
        """Читает все элементы управления и строит валидированный `InputConfig`."""
        function_key = self._selected_function_key()
        return build_input_config(
            function_key=function_key,
            kind=self._selected_value(self.kind_buttons, "max"),
            method_key=self._selected_value(self.method_buttons, "all"),
            a_raw=self.a_input.text(),
            b_raw=self.b_input.text(),
            eps_raw=self.eps_input.text(),
            l_raw=self.l_input.text(),
            coefficient_raws=self._collect_coefficient_raws(function_key),
        )

    def _set_busy(self, busy: bool) -> None:
        """Переключает окно в занятое или свободное состояние."""
        logger.info("Set busy=%s", busy)
        self.state.busy = busy
        for widget in self.control_widgets:
            widget.setEnabled(not busy)

    def on_function_change(self) -> None:
        """Переключает страницу коэффициентов под выбранный тип функции."""
        function_key = self._selected_function_key()
        index = self.function_stack_indexes.get(function_key)
        if index is not None:
            self.function_editor_stack.setCurrentIndex(index)
            current_page = self.function_editor_stack.currentWidget()
            if current_page is not None:
                self.function_editor_stack.setFixedHeight(current_page.sizeHint().height())

    def handle_run_single(self) -> None:
        """Запускает обычный расчёт по текущим параметрам пользователя."""
        logger.info("Single run requested")
        try:
            config = self._collect_input_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        self._set_busy(True)
        self.calc_task.start("Выполняется расчёт...", lambda: run_single(config))

    def handle_run_full_grid(self) -> None:
        """Запускает серию расчётов по конфигурационной сетке `ε × l`."""
        logger.info("Grid run requested")
        try:
            config = self._collect_input_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        self._set_busy(True)
        self.calc_task.start("Запускаю исследование ε × l...", lambda: run_full_grid(config))

    def _handle_worker_error(self, message: str, stack: str) -> None:
        """Обрабатывает ошибку фоновой задачи и показывает её пользователю."""
        logger.error("Worker error message=%s", message)
        logger.debug("Worker traceback\n%s", stack)
        self._set_busy(False)
        self.plot_tab.show_placeholder("Ошибка построения графика.")
        QMessageBox.critical(self, "Ошибка", message)

    def _apply_report(self, payload: object) -> None:
        """Принимает готовый `RunReport` и раскладывает его по вкладкам."""
        if not isinstance(payload, RunReport):
            return

        report = payload
        logger.info("Applying report methods=%s", report.method_keys)
        self.state.last_report = report
        self.state.selected_table_method = report.default_method_key or ""
        self.state.selected_grid_run_index = 0
        self.current_plot_version += 1

        self.summary_tab.populate(report)
        self.iterations_tab.rebuild_method_buttons(report, self.state.selected_table_method)
        self.iterations_tab.populate_grid_runs(report, self.state.selected_table_method, self.state.selected_grid_run_index)
        self._populate_iterations()
        self.plot_tab.show_placeholder("Открой вкладку «График» для автоматического построения.")
        self._set_busy(False)

        if self.tabs.currentIndex() == 2:
            self.render_last_plot()

    def _selected_grid_run(self) -> Optional[GridRunResult]:
        """Возвращает текущий выбранный прогон из серии, если он существует."""
        report = self.state.last_report
        if report is None or report.mode != "grid":
            return None
        runs = report.grid_runs_by_method.get(self.state.selected_table_method, ())
        if not runs:
            return None
        index = min(max(self.state.selected_grid_run_index, 0), len(runs) - 1)
        self.state.selected_grid_run_index = index
        return runs[index]

    def _current_iteration_result(self) -> Optional[object]:
        """Определяет, какой результат сейчас должен показываться во вкладке итераций."""
        report = self.state.last_report
        if report is None:
            return None
        if report.mode == "grid":
            run = self._selected_grid_run()
            return run.result if run is not None else None
        return report.results_by_method.get(self.state.selected_table_method)

    def _populate_iterations(self) -> None:
        """Обновляет дерево итераций под текущее состояние выбора."""
        self.iterations_tab.populate_iterations(self._current_iteration_result())

    def on_table_method_change(self) -> None:
        """Реагирует на смену выбранного метода в блоке итераций."""
        report = self.state.last_report
        if report is None:
            return
        for button in self.iterations_tab.table_method_buttons:
            if button.isChecked():
                self.state.selected_table_method = str(button.property("choice_value"))
                break
        self.state.selected_grid_run_index = 0
        self.iterations_tab.populate_grid_runs(report, self.state.selected_table_method, self.state.selected_grid_run_index)
        self._populate_iterations()
        if self.tabs.currentIndex() == 2 and report.mode == "grid":
            self.render_last_plot()

    def on_grid_run_change(self, row: int) -> None:
        """Реагирует на выбор конкретного прогона в режиме серии."""
        if row < 0:
            return
        self.state.selected_grid_run_index = row
        self._populate_iterations()
        if self.tabs.currentIndex() == 2:
            self.render_last_plot()

    def render_last_plot(self) -> None:
        """Запускает фоновое построение графика для текущего отчёта."""
        report = self.state.last_report
        if report is None:
            QMessageBox.information(self, "График", "Сначала выполните расчёт.")
            return
        if self.plot_task.is_running():
            return

        version = self.current_plot_version

        def build() -> object:
            """Собирает `Figure` в рабочем потоке по текущему состоянию окна."""
            reference_x = report.reference_point.x if report.reference_point is not None else None
            reference_f = report.reference_point.f if report.reference_point is not None else None
            if report.mode == "grid":
                runs = list(report.grid_runs_by_method.get(self.state.selected_table_method, ()))
                method_title = (
                    METHOD_SPECS[self.state.selected_table_method].title
                    if self.state.selected_table_method in METHOD_SPECS
                    else "Выбранный метод"
                )
                figure = build_grid_plot_figure(
                    function_spec=report.function_spec,
                    runs=runs,
                    selected_index=self.state.selected_grid_run_index,
                    plot_range=report.plot_range,
                    method_title=method_title,
                    reference_x=reference_x,
                    reference_f=reference_f,
                )
            else:
                plot_results = [
                    report.results_by_method[method_key]
                    for method_key in report.method_keys
                    if method_key in report.results_by_method
                ]
                figure = build_plot_figure(
                    function_spec=report.function_spec,
                    results=plot_results,
                    plot_range=report.plot_range,
                    reference_x=reference_x,
                    reference_f=reference_f,
                )
            return version, figure

        self.plot_tab.plot_state_label.setText("Строю график...")
        self.plot_task.start("Строю график...", build)

    def _finish_plot_render(self, payload: object) -> None:
        """Принимает готовую фигуру из фоновой задачи и показывает её во вкладке."""
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        version, figure = payload
        if version != self.current_plot_version or not isinstance(figure, Figure):
            return

        report = self.state.last_report
        if report is None:
            return
        self.plot_tab.show_figure(figure, self.plot_tab.context_text(report, self.state.selected_table_method, self._selected_grid_run()))

    def on_tab_changed(self, index: int) -> None:
        """Ленивая синхронизация вкладок при переключении пользователем."""
        if index == 1:
            self._populate_iterations()
        elif index == 2 and self.state.last_report is not None:
            self.render_last_plot()

    def clear_output(self) -> None:
        """Сбрасывает правую часть окна и очищает последнее состояние расчёта."""
        if self.state.busy:
            return
        logger.info("Clearing window state")
        self.state.last_report = None
        self.state.selected_table_method = ""
        self.state.selected_grid_run_index = 0
        self.summary_tab.populate(None)
        self.iterations_tab.clear()
        self.plot_tab.show_placeholder("График появится после расчёта.")


def main() -> None:
    """Создаёт `QApplication`, окно и запускает цикл обработки событий Qt."""
    logger.info("Qt GUI main start")
    app = QApplication.instance() or QApplication(sys.argv)
    window = ExtremumWindow()
    window.show()
    app.exec()
