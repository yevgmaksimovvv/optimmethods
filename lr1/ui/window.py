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
from optim_core.ui import (
    BatchRunUiController,
    BatchRunUiHooks,
    ControlsPanel,
    DarkQtThemeTokens,
    DynamicSeriesInputRow,
    ResultsPlotTabIndexes,
    TaskController,
    add_parameter_row,
    build_choice_chip_styles,
    build_dark_qt_base_styles,
    build_dynamic_series_styles,
    configure_data_table,
    configure_two_panel_splitter,
    create_choice_chip_grid,
    create_controls_panel,
    create_parameter_grid,
    create_primary_action_button,
    create_results_workspace,
    create_scroll_container,
    create_standard_group,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lr1.application.services import (
    build_input_config,
    parse_positive_series,
    run_batch,
)
from lr1.domain.functions import FUNCTION_TEMPLATE_SPECS
from lr1.domain.models import AppState, GridRunResult, InputConfig, RunReport
from lr1.domain.search import METHOD_SPECS
from lr1.infrastructure.logging import configure_logging, get_log_file_path
from lr1.infrastructure.settings import DEFAULT_INPUT_EPS, DEFAULT_INTERVAL, DEFAULT_L
from lr1.ui.plotting import build_grid_plot_figure, build_plot_figure
from lr1.ui.tabs import IterationsTab, PlotTab, SummaryTab

configure_logging()
logger = logging.getLogger("lr1.gui")

APP_TITLE = "ЛР1 — Методы одномерной оптимизации"
CONTROL_BUTTON_SIZE = 44
SERIES_INPUT_WIDTH = 96
ROW_CONTROL_SPACING = 4


class ExtremumWindow(QMainWindow):
    """Основное окно лабораторного приложения."""
    def __init__(self):
        """Создаёт окно, фоновые контроллеры и базовое состояние интерфейса."""
        super().__init__()
        logger.info("ExtremumWindow init start")
        self.state = AppState()
        self.current_plot_version = 0
        self.control_widgets: List[QWidget] = []
        self.coefficient_tables: Dict[str, QTableWidget] = {}
        self.coefficient_keys: Dict[str, tuple[str, ...]] = {}
        self.function_stack_indexes: Dict[str, int] = {}
        self.tab_indexes = ResultsPlotTabIndexes(results=0, plot=1)
        self.epsilon_row: DynamicSeriesInputRow | None = None
        self.l_row: DynamicSeriesInputRow | None = None
        self._run_flow = BatchRunUiController[RunReport](
            BatchRunUiHooks(
                assign_report=self._assign_report_state,
                reset_selection=self._reset_report_selection,
                render_overview=self._render_report_overview,
                select_first=self._select_first_after_run,
                clear_details=self._clear_report_details,
            )
        )

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
            build_dark_qt_base_styles(
                DarkQtThemeTokens(
                    background="#1b1f2a",
                    text="#f0f2f5",
                    font_family='"Segoe UI", "Helvetica Neue", "Arial", sans-serif',
                    group_border="#3f4a62",
                    group_radius_px=12,
                    group_padding_px=13,
                    group_title_color="#dde5f3",
                    button_bg="#2b3447",
                    button_border="#4b5873",
                    button_hover_bg="#36415a",
                    button_pressed_bg="#242d3d",
                    button_disabled_bg="#1f2533",
                    button_disabled_text="#66738d",
                    button_disabled_border="#3c465f",
                    primary_bg="#0f7aff",
                    primary_border="#3b94ff",
                    primary_hover_bg="#2588ff",
                    primary_pressed_bg="#0d66d8",
                    tab_bg="#2c303b",
                    tab_border="#4b4f5c",
                    tab_selected_bg="#46516b",
                    tab_selected_border="#647596",
                )
            )
            + """
            QPushButton[role="action"] {
                padding: 8px 16px;
                min-height: 22px;
                font-size: 15px;
            }
            QListWidget {
                background: #131824;
                border: 1px solid #3f4a62;
                border-radius: 8px;
                padding: 7px 10px;
                color: #f5f7fb;
                selection-background-color: #2379ff;
            }
            QListWidget#GridRunList {
                background: transparent;
                border: 0;
                padding: 0;
            }
            QTreeWidget {
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
            QListWidget#GridRunList::item:selected {
                background: transparent;
            }
            QWidget#GridRunCard {
                background: #171b24;
                border: 1px solid #31384a;
                border-radius: 12px;
            }
            QListWidget#GridRunList::item:selected QWidget#GridRunCard {
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
            QListWidget#GridRunList::item:selected QWidget#GridRunCard QLabel[role="grid-run-metric-caption"] {
                color: #cddcff;
            }
            QListWidget#GridRunList::item:selected QWidget#GridRunCard QLabel[role="grid-run-metric-value"] {
                color: #ffffff;
            }
            QTreeWidget#IterationsTree {
                background: #12161d;
                alternate-background-color: #151b25;
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
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
            QLabel#SectionHint {
                font-size: 14px;
                padding: 4px 2px;
            }
            QWidget#FinalResultCard {
                background: #181b24;
                border: 1px solid #31384a;
                border-radius: 12px;
                min-height: 132px;
            }
            QLabel#FinalResultTitle {
                color: #d5e3ff;
                font-size: 13px;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            QLabel[role="final-result-caption"] {
                color: #9aa5bb;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel[role="final-result-value"] {
                color: #eef2f8;
                font-size: 16px;
                font-weight: 700;
            }
            QWidget#FunctionFormulaCard, QWidget#CoefficientCard {
                background: #181b24;
                border: 1px solid #31384a;
                border-radius: 12px;
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
            """
            + build_choice_chip_styles()
            + build_dynamic_series_styles()
        )

    def _build_ui(self) -> None:
        """Собирает двухпанельный интерфейс: управление слева, результаты справа."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        controls: ControlsPanel = create_controls_panel(min_width=500, max_width=560, spacing=12)
        left_panel = controls.panel
        left_layout = controls.layout

        self._build_function_box(left_layout)
        self._build_scenario_box(left_layout)
        self._build_parameters_box(left_layout)
        self._build_actions_box(left_layout)

        left_layout.addStretch(1)
        left_scroll = create_scroll_container(
            left_panel,
            widget_resizable=True,
            horizontal_policy=Qt.ScrollBarAlwaysOff,
        )

        workspace = create_results_workspace(
            results_title="Таблицы",
            plot_title="Графики",
            with_tables_empty_state=True,
            tables_empty_title="Пока нет результатов",
            tables_empty_description=(
                "Слева выбери функцию, диапазон и метод.\n"
                "После запуска здесь появятся результаты и сравнение методов."
            ),
            tables_empty_hint="Нажми «Рассчитать», чтобы получить результаты и графики.",
        )
        self.tabs = workspace.tabs
        self.results_tab_stack = workspace.tables_empty_stack
        if self.results_tab_stack is None:
            raise RuntimeError("Ожидался EmptyStateStack для вкладки таблиц")

        self.summary_tab = SummaryTab()
        self.iterations_tab = IterationsTab(self.on_grid_run_change, self.on_table_method_change)
        self.plot_tab = PlotTab()

        workspace.tables_layout.addWidget(self.summary_tab)
        workspace.tables_layout.addWidget(self.iterations_tab)
        workspace.plots_layout.addWidget(self.plot_tab)
        self.tab_indexes = workspace.tab_indexes
        self.tabs.currentChanged.connect(self.on_tab_changed)

        configure_two_panel_splitter(
            splitter,
            left=left_scroll,
            right=workspace.panel,
            left_size=540,
            right_size=960,
        )

    def _build_function_box(self, layout: QVBoxLayout) -> None:
        """Строит секцию выбора типа функции и редактирования коэффициентов."""
        box, box_layout = create_standard_group("Функция")

        self.function_group = QButtonGroup(self)
        self.function_group.setExclusive(True)
        selector_holder, self.function_buttons = create_choice_chip_grid(
            group=self.function_group,
            options=(("Квадратичная", "quadratic"), ("Рациональная", "rational")),
            columns=2,
            horizontal_spacing=6,
            vertical_spacing=6,
        )
        for button in self.function_buttons:
            button.toggled.connect(self.on_function_change)
        box_layout.addWidget(selector_holder)

        self._build_function_editors(box_layout)
        layout.addWidget(box)

    def _build_scenario_box(self, layout: QVBoxLayout) -> None:
        """Строит блок выбора цели поиска и алгоритма."""
        box = QGroupBox("Сценарий")
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        grid = QGridLayout(box)
        grid.setContentsMargins(16, 16, 16, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
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
        box = QGroupBox("Параметры расчёта")
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        grid = create_parameter_grid(box, vertical_spacing=6)

        self.a_input, self.b_input = self._add_interval_inputs(grid, row=0)
        self._add_series_inputs(grid, row=1, label="Точности ε", target="eps")
        self._add_series_inputs(grid, row=2, label="Длины интервала l", target="l")

        layout.addWidget(box)

    def _build_actions_box(self, layout: QVBoxLayout) -> None:
        """Строит блок запуска расчёта."""
        self.run_button = create_primary_action_button(text="Рассчитать", on_click=self.handle_run)
        layout.addWidget(self.run_button)

        self.control_widgets.extend(
            [
                self.a_input,
                self.b_input,
                self.run_button,
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
    ) -> List[QAbstractButton]:
        """Создаёт сетку радиокнопок и возвращает список созданных элементов."""
        base_row = row * 2
        label_widget = QLabel(label)
        label_widget.setProperty("role", "parameter-label")
        label_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(label_widget, base_row, 0, 1, columns, Qt.AlignLeft | Qt.AlignVCenter)
        holder, buttons = create_choice_chip_grid(
            group=group,
            options=options,
            columns=columns,
            horizontal_spacing=10,
            vertical_spacing=8,
        )
        grid.addWidget(holder, base_row + 1, 0, 1, columns)
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

            coeff_table = QTableWidget(1, len(template.coefficients))
            coeff_table.setAlternatingRowColors(True)
            coeff_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            coeff_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
            coeff_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            coeff_table.horizontalHeader().setMinimumSectionSize(72)
            coeff_table.horizontalHeader().setFixedHeight(34)
            coeff_table.verticalHeader().setVisible(False)
            coeff_table.verticalHeader().setDefaultSectionSize(42)
            coeff_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            coeff_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            coeff_table.setFixedHeight(94)
            coeff_table.setProperty("variant", "report")
            coeff_table.setHorizontalHeaderLabels([item.label for item in template.coefficients])
            configure_data_table(
                coeff_table,
                min_row_height=42,
                allow_selection=False,
                allow_editing=True,
                word_wrap=False,
            )

            self.coefficient_tables[template.key] = coeff_table
            self.coefficient_keys[template.key] = tuple(item.key for item in template.coefficients)
            self.control_widgets.append(coeff_table)

            for column, coefficient in enumerate(template.coefficients):
                item = QTableWidgetItem(f"{coefficient.default:g}")
                item.setTextAlignment(Qt.AlignCenter)
                coeff_table.setItem(0, column, item)

            coeff_card_layout.addWidget(coeff_table)
            page_layout.addWidget(coeff_card)
            self.function_editor_stack.addWidget(page)
            self.function_stack_indexes[template.key] = index

    def _add_interval_inputs(self, grid: QGridLayout, row: int) -> tuple[QLineEdit, QLineEdit]:
        """Добавляет в сетку пару полей для левой и правой границы интервала."""
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        left_caption = QLabel("a")
        left_caption.setObjectName("CoeffCaption")
        left_caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(left_caption)

        left_line = QLineEdit()
        left_line.setMinimumWidth(170)
        left_line.setMinimumHeight(38)
        left_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(left_line, 1)

        right_caption = QLabel("b")
        right_caption.setObjectName("CoeffCaption")
        right_caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(right_caption)

        right_line = QLineEdit()
        right_line.setMinimumWidth(170)
        right_line.setMinimumHeight(38)
        right_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(right_line, 1)

        add_parameter_row(grid, row=row, label="Границы", control=holder)
        return left_line, right_line

    def _add_series_inputs(self, grid: QGridLayout, row: int, label: str, target: str) -> None:
        """Добавляет динамический ряд параметров с кнопками `+` и `−`."""
        row_control = DynamicSeriesInputRow(
            add_role="series-add",
            remove_role="series-remove",
            field_role="series-item",
            placeholders=("ε",) if target == "eps" else ("L",),
            field_widths=(SERIES_INPUT_WIDTH,),
            control_button_size=CONTROL_BUTTON_SIZE,
            row_control_spacing=ROW_CONTROL_SPACING,
            on_control_added=self.control_widgets.append,
            on_control_removed=self._remove_control_widget,
        )
        add_parameter_row(grid, row=row, label=label, control=row_control.row_widget)
        row_control.add_item(DEFAULT_INPUT_EPS if target == "eps" else DEFAULT_L)

        if target == "eps":
            self.epsilon_row = row_control
        else:
            self.l_row = row_control

    def _set_defaults(self) -> None:
        """Заполняет интерфейс начальными значениями по умолчанию."""
        self._reset_function_defaults()
        self.function_buttons[0].setChecked(True)
        self.kind_buttons[0].setChecked(True)
        self.method_buttons[0].setChecked(True)
        self.a_input.setText(DEFAULT_INTERVAL[0])
        self.b_input.setText(DEFAULT_INTERVAL[1])
        if self.epsilon_row is not None:
            self.epsilon_row.reset(((DEFAULT_INPUT_EPS,),))
        if self.l_row is not None:
            self.l_row.reset(((DEFAULT_L,),))
        self.on_function_change()
        self.summary_tab.populate(None)

    def _reset_function_defaults(self) -> None:
        """Возвращает коэффициенты всех шаблонов функций к исходным значениям."""
        for template in FUNCTION_TEMPLATE_SPECS.values():
            table = self.coefficient_tables.get(template.key)
            if table is None:
                continue
            for column, coefficient in enumerate(template.coefficients):
                item = table.item(0, column)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(0, column, item)
                item.setText(f"{coefficient.default:g}")

    def _selected_value(self, buttons: List[QAbstractButton], default: str) -> str:
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
        table = self.coefficient_tables.get(function_key)
        keys = self.coefficient_keys.get(function_key, ())
        if table is None or not keys:
            return {}
        values: Dict[str, str] = {}
        for column, key in enumerate(keys):
            item = table.item(0, column)
            values[key] = item.text().strip() if item is not None else ""
        return values

    def _collect_run_request(self) -> tuple[InputConfig, tuple[float, ...], tuple[float, ...]]:
        """Читает элементы управления и готовит полный запрос на пакетный запуск."""
        function_key = self._selected_function_key()
        eps_values = parse_positive_series(self._collect_series_raw(self.epsilon_row, "Список ε пуст."), "ε")
        l_values = parse_positive_series(self._collect_series_raw(self.l_row, "Список l пуст."), "l")
        config = build_input_config(
            function_key=function_key,
            kind=self._selected_value(self.kind_buttons, "max"),
            method_key=self._selected_value(self.method_buttons, "all"),
            a_raw=self.a_input.text(),
            b_raw=self.b_input.text(),
            eps_raw=str(eps_values[0]),
            l_raw=str(l_values[0]),
            coefficient_raws=self._collect_coefficient_raws(function_key),
        )
        return config, eps_values, l_values

    def _collect_series_raw(self, row: DynamicSeriesInputRow | None, empty_message: str) -> str:
        if row is None:
            raise ValueError(empty_message)
        values = [field_values[0] for field_values in row.rows() if field_values and field_values[0]]
        if not values:
            raise ValueError(empty_message)
        return ",".join(values)

    def _remove_control_widget(self, widget: QWidget) -> None:
        if widget in self.control_widgets:
            self.control_widgets.remove(widget)

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

    def handle_run(self) -> None:
        """Запускает расчёт по всем комбинациям введённых `ε × l`."""
        logger.info("Run requested")
        try:
            config, eps_values, l_values = self._collect_run_request()
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        self._set_busy(True)
        self.calc_task.start(
            "Выполняется расчёт...",
            lambda: run_batch(config, eps_values=eps_values, l_values=l_values),
        )

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
        self._run_flow.apply(report)
        self._set_busy(False)

        if self.tabs.currentIndex() == self.tab_indexes.plot:
            self.render_last_plot()

    def _assign_report_state(self, report: RunReport) -> None:
        self.state.last_report = report

    def _reset_report_selection(self) -> None:
        report = self.state.last_report
        if report is None:
            self.state.selected_table_method = ""
            self.state.selected_grid_run_index = 0
            return
        self.state.selected_table_method = report.default_method_key or ""
        self.state.selected_grid_run_index = 0
        self.current_plot_version += 1

    def _render_report_overview(self, report: RunReport) -> None:
        self.results_tab_stack.set_empty(False)
        self.summary_tab.populate(report)
        self.iterations_tab.rebuild_method_buttons(report, self.state.selected_table_method)
        self.iterations_tab.populate_grid_runs(report, self.state.selected_table_method, self.state.selected_grid_run_index)
        self._populate_iterations()
        self.plot_tab.show_placeholder("Открой вкладку «График» для автоматического построения.")

    def _select_first_after_run(self, report: RunReport) -> bool:
        return bool(report.method_keys)

    def _clear_report_details(self) -> None:
        self.results_tab_stack.set_empty(True)
        self.iterations_tab.clear()
        self.plot_tab.show_placeholder("График появится после расчёта.")

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
        if self.tabs.currentIndex() == self.tab_indexes.plot and report.mode == "grid":
            self.render_last_plot()

    def on_grid_run_change(self, row: int) -> None:
        """Реагирует на выбор конкретного прогона в режиме серии."""
        if row < 0:
            return
        self.state.selected_grid_run_index = row
        self._populate_iterations()
        if self.tabs.currentIndex() == self.tab_indexes.plot:
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
        if index == self.tab_indexes.results:
            self._populate_iterations()
        elif index == self.tab_indexes.plot and self.state.last_report is not None:
            self.render_last_plot()

def main() -> None:
    """Создаёт `QApplication`, окно и запускает цикл обработки событий Qt."""
    logger.info("Qt GUI main start")
    app = QApplication.instance() or QApplication(sys.argv)
    window = ExtremumWindow()
    window.show()
    app.exec()
