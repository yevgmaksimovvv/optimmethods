"""GUI для ЛР5: метод барьерных функций поверх итераций Розенброка из ЛР2."""

from __future__ import annotations

import html
import math
import sys
import uuid
from dataclasses import dataclass

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSplitter,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from optim_core.parsing import parse_localized_float
from optim_core.ui import (
    ControlsPanel,
    DarkQtThemeTokens,
    EmptyStateStack,
    MathHeaderView,
    PlotCanvas,
    TaskController,
    add_parameter_row,
    build_choice_chip_styles,
    build_dark_qt_base_styles,
    clear_plot_canvas,
    configure_data_table,
    configure_two_panel_splitter,
    create_choice_chip_grid,
    create_controls_panel,
    create_flush_row,
    create_parameter_grid,
    create_primary_action_button,
    create_results_workspace,
    create_scroll_container,
    create_standard_group,
    dark_plot_context,
    set_table_data_layout,
)

from lr5.application.artifacts import BarrierArtifactsStore
from lr5.application.services import build_method_config, parse_vector, run_barrier_method, variant_2_problem
from lr5.domain.barrier import is_strictly_feasible
from lr5.domain.models import BarrierIterationResult, BarrierResult

APP_TITLE = "ЛР5 — Метод барьерных функций"
DEFAULT_START_POINT = "0;0"
DEFAULT_MU0_TEXT = "10"
DEFAULT_BETA_TEXT = "0.1"
DEFAULT_EPSILON_OUTER_TEXT = "1e-3"
DEFAULT_MAX_OUTER_ITERATIONS_TEXT = "20"
DEFAULT_INNER_EPSILON = "1e-4"
DEFAULT_INNER_MAX_ITERATIONS = "200"


@dataclass(frozen=True)
class RunState:
    """Результат запуска, который нужен UI для сохранения и отображения."""

    result: BarrierResult
    trace_id: str


class BarrierWindow(QMainWindow):
    """Главное окно ЛР5."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1530, 960)
        self._apply_styles()

        self._problem = variant_2_problem()
        self._run_state: RunState | None = None
        self._selected_iteration_index: int | None = None
        self._artifacts_store = BarrierArtifactsStore()
        self._run_task = TaskController(self)
        self._run_task.succeeded.connect(self._on_run_succeeded)
        self._run_task.failed.connect(self._on_run_failed)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        self._controls_panel = self._build_controls_panel()
        controls_scroll = create_scroll_container(
            self._controls_panel,
            widget_resizable=True,
            horizontal_policy=Qt.ScrollBarAlwaysOff,
        )
        results = self._build_results_panel()

        configure_two_panel_splitter(
            splitter,
            left=controls_scroll,
            right=results,
            left_size=510,
            right_size=1020,
            handle_width=8,
        )

        self.barrier_buttons["reciprocal"].setChecked(True)
        self._on_barrier_selected("reciprocal", True)
        self._reset_view()

    def _apply_styles(self) -> None:
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
            + build_choice_chip_styles()
            + """
            QLineEdit {
                min-height: 26px;
                font-size: 12px;
                padding: 5px 9px;
            }
            QLabel#SectionCaption {
                font-size: 10px;
            }
            QLabel[role="parameter-label"] {
                font-size: 13px;
                font-weight: 600;
            }
            QLabel[role="formula-preview"] {
                min-height: 88px;
                text-align: center;
                padding: 10px 12px;
                font-family: "Cambria Math", "Times New Roman", serif;
                font-size: 14px;
                font-weight: 700;
                background: #101722;
                border: 1px solid #31425a;
                border-radius: 10px;
            }
            QWidget#SummaryMetricCard {
                background: #101722;
                border: 1px solid #2f3d56;
                border-radius: 10px;
            }
            QLabel[role="summary-caption"] {
                color: #9ca8bc;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel[role="summary-value"] {
                color: #f2f5fa;
                font-size: 13px;
                font-weight: 700;
                line-height: 1.3;
            }
            QLabel[role="summary-status"] {
                color: #eff5ff;
                font-size: 11px;
                font-weight: 800;
                line-height: 1.25;
                padding: 5px 10px;
                background: #142236;
                border: 1px solid #2f4664;
                border-radius: 999px;
            }
            QLabel[role="summary-comment"] {
                color: #dbe5f4;
                font-size: 12px;
                line-height: 1.35;
            }
            QLabel[role="summary-constraints"] {
                color: #aeb9ca;
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
                font-size: 11px;
                line-height: 1.35;
            }
            QLabel[role="summary-conclusion"] {
                color: #dce6f5;
                font-size: 12px;
                font-weight: 600;
                line-height: 1.35;
            }
            QLabel[role="summary-note"] {
                color: #d8e2f2;
                font-size: 12px;
                line-height: 1.35;
                padding: 10px 12px;
                background: #101722;
                border: 1px solid #2f3d56;
                border-radius: 10px;
            }
            QLabel[role="status"] {
                padding: 10px 12px;
                background: #101722;
                border: 1px solid #31425a;
                border-radius: 10px;
                min-height: 54px;
                line-height: 1.25;
            }
            QPushButton[variant="tertiary"] {
                min-height: 28px;
                padding: 4px 10px;
                color: #dce6f5;
                background: #242d3d;
                border: 1px solid #3c4861;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton[variant="tertiary"]:hover {
                background: #2d3850;
            }
            QPushButton[variant="tertiary"]:pressed {
                background: #1f2736;
            }
            QTableWidget {
                background: #10141f;
                alternate-background-color: #141927;
            }
            QGroupBox#SummaryCard,
            QGroupBox#IterationCard,
            QGroupBox#InnerHistoryCard,
            QGroupBox#AdvancedCard {
                padding-top: 12px;
            }
            QGroupBox#AdvancedCard {
                color: #b7c4d8;
                border-color: #334055;
            }
            QGroupBox#AdvancedCard::title {
                color: #b7c4d8;
                font-size: 11px;
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
            QToolButton[role="section-toggle"] {
                color: #dde5f3;
                background: transparent;
                border: 0;
                font-size: 12px;
                font-weight: 700;
                padding: 3px 4px 5px 4px;
                min-height: 26px;
            }
            """
        )

    def _build_controls_panel(self) -> QWidget:
        controls: ControlsPanel = create_controls_panel(min_width=500, max_width=560, spacing=7)
        panel = controls.panel
        layout = controls.layout

        basic_group, basic_layout = create_standard_group("Основные", spacing=4, margins=(12, 12, 12, 10))
        problem_group, problem_layout = create_standard_group("Постановка задачи", spacing=4, margins=(12, 12, 12, 10))
        problem_caption = QLabel()
        problem_caption.setProperty("role", "formula-preview")
        problem_caption.setTextFormat(Qt.RichText)
        problem_caption.setWordWrap(True)
        problem_caption.setText(
            "<b>F(x<sub>1</sub>, x<sub>2</sub>) = (x<sub>1</sub> - 5)<sup>2</sup> + "
            "(x<sub>2</sub> - 3)<sup>2</sup></b><br>"
            "g<sub>1</sub>(x) = -x<sub>1</sub> + 2x<sub>2</sub> - 4 &le; 0<br>"
            "g<sub>2</sub>(x) = x<sub>1</sub> + x<sub>2</sub> - 3 &le; 0"
        )
        problem_layout.addWidget(problem_caption)
        basic_layout.addWidget(problem_group)

        start_point_group = QGroupBox("Стартовая точка")
        start_point_grid = create_parameter_grid(start_point_group, label_min_width=88)
        self.start_x1_input = QLineEdit()
        self.start_x2_input = QLineEdit()
        self.start_x1_input.setPlaceholderText("x1")
        self.start_x2_input.setPlaceholderText("x2")
        self.start_x1_input.setText(DEFAULT_START_POINT.split(";")[0])
        self.start_x2_input.setText(DEFAULT_START_POINT.split(";")[1])
        start_row = QWidget()
        start_row_layout = QHBoxLayout(start_row)
        start_row_layout.setContentsMargins(0, 0, 0, 0)
        start_row_layout.setSpacing(6)
        start_x1_caption = QLabel("x₁")
        start_x1_caption.setObjectName("SectionCaption")
        start_x2_caption = QLabel("x₂")
        start_x2_caption.setObjectName("SectionCaption")
        start_row_layout.addWidget(start_x1_caption)
        start_row_layout.addWidget(self.start_x1_input, 1)
        start_row_layout.addWidget(start_x2_caption)
        start_row_layout.addWidget(self.start_x2_input, 1)
        start_point_grid.addWidget(start_row, 0, 0, 1, 3)
        basic_layout.addWidget(start_point_group)

        outer_group = QGroupBox("Параметры внешнего цикла")
        outer_grid = create_parameter_grid(outer_group)
        self.mu0_input = QLineEdit(DEFAULT_MU0_TEXT)
        self.beta_input = QLineEdit(DEFAULT_BETA_TEXT)
        self.outer_epsilon_input = QLineEdit(DEFAULT_EPSILON_OUTER_TEXT)
        self.outer_max_iterations_input = QLineEdit(DEFAULT_MAX_OUTER_ITERATIONS_TEXT)
        self.mu0_input.setPlaceholderText("10")
        self.beta_input.setPlaceholderText("0.1")
        self.outer_epsilon_input.setPlaceholderText("1e-3")
        self.outer_max_iterations_input.setPlaceholderText("20")
        add_parameter_row(outer_grid, row=0, label="Начальное μ", control=self.mu0_input)
        add_parameter_row(outer_grid, row=1, label="Коэффициент уменьшения β", control=self.beta_input)
        add_parameter_row(outer_grid, row=2, label="Точность внешнего цикла", control=self.outer_epsilon_input)
        add_parameter_row(outer_grid, row=3, label="Макс. число внешних итераций", control=self.outer_max_iterations_input)
        basic_layout.addWidget(outer_group)

        barrier_group, barrier_layout = create_standard_group("Вид барьера", spacing=4, margins=(12, 12, 12, 10))
        self.barrier_button_group = QButtonGroup(self)
        self.barrier_button_group.setExclusive(True)
        barrier_row, barrier_buttons = create_choice_chip_grid(
            group=self.barrier_button_group,
            options=(("Обратный", "reciprocal"), ("Логарифмический", "log")),
            columns=2,
            horizontal_spacing=4,
            vertical_spacing=4,
            on_clicked=self._on_barrier_selected,
            tooltips={
                "reciprocal": "Обратный барьер: -1/g(x), ближе к границе растёт резче.",
                "log": "Логарифмический барьер: -log(-g(x)), мягче ведёт себя у границы.",
            },
        )
        self.barrier_buttons = {value: button for value, button in zip(("reciprocal", "log"), barrier_buttons, strict=True)}
        barrier_layout.addWidget(barrier_row)
        basic_layout.addWidget(barrier_group)

        advanced_group = QWidget()
        advanced_group.setObjectName("AdvancedCard")
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(0)
        advanced_header, self.advanced_toggle = self._build_collapsible_header("Дополнительные")
        advanced_layout.addWidget(advanced_header)
        advanced_content = QWidget()
        advanced_grid = create_parameter_grid(advanced_content, margins=(14, 0, 14, 14))
        self.inner_epsilon_input = QLineEdit(DEFAULT_INNER_EPSILON)
        self.inner_max_iterations_input = QLineEdit(DEFAULT_INNER_MAX_ITERATIONS)
        add_parameter_row(
            advanced_grid,
            row=0,
            label="Точность внутренних итераций",
            control=self.inner_epsilon_input,
        )
        add_parameter_row(
            advanced_grid,
            row=1,
            label="Макс. итераций",
            control=self.inner_max_iterations_input,
        )
        advanced_layout.addWidget(advanced_content)
        self._bind_toggle(self.advanced_toggle, advanced_content, expanded=False)

        actions_group, actions_layout = create_standard_group("Управление", spacing=6, margins=(12, 12, 12, 10))
        run_button = create_primary_action_button(text="Рассчитать", on_click=self._on_run_clicked)
        reset_button = create_primary_action_button(text="Сбросить", on_click=self._reset_inputs, role="secondary", min_height=34)
        self.run_button = run_button
        self.reset_button = reset_button
        buttons_row, _buttons_layout = create_flush_row(run_button, reset_button)
        actions_layout.addWidget(buttons_row)

        layout.addWidget(basic_group)
        layout.addWidget(advanced_group)
        layout.addWidget(actions_group)

        self.status_label = QLabel("Готов к запуску.")
        self.status_label.setProperty("role", "status")
        self.status_label.setWordWrap(True)

        layout.addStretch(1)
        return panel

    def _build_results_panel(self) -> QWidget:
        workspace = create_results_workspace(
            results_title="Отчёт",
            plot_title="График",
            with_tables_empty_state=True,
            tables_empty_title="Пока нет результатов",
            tables_empty_description=(
                "Слева задай задачу, стартовую точку и параметры внешнего цикла.\n"
                "После расчёта здесь появятся итог, внешние итерации и график."
            ),
            tables_empty_hint="Нажми «Рассчитать», чтобы увидеть отчёт и график.",
        )
        self.results_tabs = workspace.tabs
        self.results_tab_indexes = workspace.tab_indexes
        self.report_empty_stack = workspace.tables_empty_stack
        tables_layout = workspace.tables_layout

        self.report_scroll = QScrollArea()
        self.report_scroll.setWidgetResizable(True)
        self.report_scroll.setFrameShape(QScrollArea.NoFrame)
        self.report_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.report_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.report_scroll.verticalScrollBar().setSingleStep(32)

        self.report_content = QWidget()
        self.report_layout = QVBoxLayout(self.report_content)
        self.report_layout.setContentsMargins(0, 0, 0, 0)
        self.report_layout.setSpacing(12)
        self.report_scroll.setWidget(self.report_content)
        tables_layout.addWidget(self.report_scroll)

        save_row = QWidget()
        save_row_layout = QHBoxLayout(save_row)
        save_row_layout.setContentsMargins(0, 0, 0, 0)
        save_row_layout.addStretch(1)
        self.save_button = create_primary_action_button(text="Сохранить результаты", on_click=self._save_result, role="secondary", min_height=30)
        self.save_button.setProperty("variant", "tertiary")
        self.save_button.setEnabled(False)
        save_row_layout.addWidget(self.save_button)
        self.report_layout.addWidget(save_row)

        summary_group = QGroupBox("Итог")
        summary_group.setObjectName("SummaryCard")
        summary_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setContentsMargins(16, 12, 16, 12)
        summary_layout.setSpacing(8)

        summary_header = QWidget()
        summary_header_layout = QHBoxLayout(summary_header)
        summary_header_layout.setContentsMargins(0, 0, 0, 0)
        summary_header_layout.setSpacing(10)

        self.summary_status = QLabel("Пока нет результатов")
        self.summary_status.setProperty("role", "summary-status")
        self.summary_status.setTextFormat(Qt.PlainText)
        self.summary_status.setWordWrap(False)
        summary_header_layout.addWidget(self.summary_status)

        self.summary_comment = QLabel("Задай параметры слева и нажми «Рассчитать».")
        self.summary_comment.setProperty("role", "summary-comment")
        self.summary_comment.setTextFormat(Qt.PlainText)
        self.summary_comment.setWordWrap(True)
        summary_header_layout.addWidget(self.summary_comment, 1)
        summary_layout.addWidget(summary_header)

        summary_metrics = QWidget()
        summary_metrics_layout = QGridLayout(summary_metrics)
        summary_metrics_layout.setContentsMargins(0, 0, 0, 0)
        summary_metrics_layout.setHorizontalSpacing(8)
        summary_metrics_layout.setVerticalSpacing(8)
        self._summary_metric_cards: list[QWidget] = []

        point_card, self.summary_point_caption, self.summary_point = self._build_summary_metric_card("Итоговая точка", "—")
        value_card, self.summary_value_caption, self.summary_value = self._build_summary_metric_card("Значение F(x)", "—")
        barrier_card, self.summary_barrier_caption, self.summary_barrier = self._build_summary_metric_card("Вид барьера", "—")
        mu_card, self.summary_mu_caption, self.summary_mu = self._build_summary_metric_card("Последнее μ", "—")
        self._summary_metric_cards.extend([point_card, value_card, barrier_card, mu_card])

        summary_metrics_layout.addWidget(point_card, 0, 0)
        summary_metrics_layout.addWidget(value_card, 0, 1)
        summary_metrics_layout.addWidget(barrier_card, 1, 0)
        summary_metrics_layout.addWidget(mu_card, 1, 1)
        summary_layout.addWidget(summary_metrics)

        summary_footer = QWidget()
        summary_footer_layout = QHBoxLayout(summary_footer)
        summary_footer_layout.setContentsMargins(0, 0, 0, 0)
        summary_footer_layout.setSpacing(12)
        self.summary_constraints = QLabel("g1(x) = —\ng2(x) = —")
        self.summary_constraints.setProperty("role", "summary-constraints")
        self.summary_constraints.setTextFormat(Qt.PlainText)
        self.summary_constraints.setWordWrap(True)
        summary_footer_layout.addWidget(self.summary_constraints, 1)

        self.summary_conclusion = QLabel("Пока нет результата.")
        self.summary_conclusion.setProperty("role", "summary-conclusion")
        self.summary_conclusion.setTextFormat(Qt.PlainText)
        self.summary_conclusion.setWordWrap(True)
        summary_footer_layout.addWidget(self.summary_conclusion, 1)
        summary_layout.addWidget(summary_footer)

        self.report_layout.addWidget(summary_group)

        outer_group, outer_layout = create_standard_group("Внешние итерации", spacing=6, margins=(14, 12, 14, 12))
        outer_column_count = 8 + len(self._problem.constraints)
        self.outer_table = QTableWidget(0, outer_column_count)
        configure_data_table(self.outer_table, min_row_height=30, allow_selection=True, allow_editing=False, word_wrap=False)
        self.outer_table.verticalHeader().setVisible(False)
        outer_header = MathHeaderView(Qt.Horizontal, self.outer_table)
        outer_header.set_math_labels([
            "k",
            "μ<sub>k</sub>",
            "x<sub>μ_k</sub>",
            "F(x)",
            "M(x)",
            "μ<sub>k</sub>·M(x)",
            "Θ(x)",
            *[f"g<sub>{index + 1}</sub>(x)" for index in range(len(self._problem.constraints))],
            "N<sub>inner</sub>",
        ])
        self.outer_table.setHorizontalHeader(outer_header)
        self.outer_table.itemSelectionChanged.connect(self._on_outer_selection_changed)
        outer_layout.addWidget(self.outer_table)
        self.report_layout.addWidget(outer_group)

        inner_group = QWidget()
        inner_group.setObjectName("InnerHistoryCard")
        inner_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        inner_layout = QVBoxLayout(inner_group)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(0)
        inner_header, self.inner_details_toggle = self._build_collapsible_header("Подробности внутренних итераций")
        inner_layout.addWidget(inner_header)
        inner_content = QWidget()
        inner_content_layout = QVBoxLayout(inner_content)
        inner_content_layout.setContentsMargins(14, 12, 14, 12)
        inner_content_layout.setSpacing(8)

        self.inner_label = QLabel("Выберите внешнюю итерацию, чтобы увидеть внутреннюю историю.")
        self.inner_label.setWordWrap(True)
        self.inner_label.setProperty("role", "summary-value")
        inner_content_layout.addWidget(self.inner_label)

        self.inner_table = QTableWidget(0, 10)
        configure_data_table(self.inner_table, min_row_height=30, allow_selection=False, allow_editing=False, word_wrap=False)
        self.inner_table.verticalHeader().setVisible(False)
        self.inner_table.setHorizontalHeaderLabels(["K", "x_k", "F(x_k)", "j", "d_j", "y_j", "f(y_j)", "lambda_j", "y_{j+1}", "f(y_{j+1})"])
        self.inner_table.setMaximumHeight(220)
        inner_content_layout.addWidget(self.inner_table)
        inner_layout.addWidget(inner_content)
        self._bind_toggle(self.inner_details_toggle, inner_content, expanded=False)
        self.report_layout.addWidget(inner_group)
        self.report_layout.addStretch(1)

        plot_tab_content = QWidget()
        plot_tab_layout = QVBoxLayout(plot_tab_content)
        plot_tab_layout.setContentsMargins(0, 0, 0, 0)
        plot_tab_layout.setSpacing(12)

        self.canvas = PlotCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(620)
        plot_tab_layout.addWidget(self.canvas)
        self.plot_empty_stack = EmptyStateStack(
            title="Пока нет графика",
            description=(
                "После расчёта здесь появятся допустимая область, ограничения, стартовая точка, "
                "внешняя траектория и итоговое положение."
            ),
            hint="Нажми «Рассчитать», чтобы построить график и увидеть траекторию.",
            content_widget=plot_tab_content,
        )
        workspace.plots_layout.addWidget(self.plot_empty_stack)

        return workspace.panel

    def _on_barrier_selected(self, value: str, checked: bool) -> None:
        if not checked:
            return
        for key, button in self.barrier_buttons.items():
            button.setChecked(key == value)

    def _selected_barrier_kind(self) -> str:
        for key, button in self.barrier_buttons.items():
            if button.isChecked():
                return key
        return "reciprocal"

    def _on_run_clicked(self) -> None:
        if self._run_task.is_running():
            self._set_status("Расчёт уже выполняется.", kind="error")
            return

        try:
            start_point = parse_vector(f"{self.start_x1_input.text().strip()};{self.start_x2_input.text().strip()}")
            mu0 = parse_localized_float(self.mu0_input.text().strip(), "mu0")
            beta = parse_localized_float(self.beta_input.text().strip(), "beta")
            outer_epsilon = parse_localized_float(self.outer_epsilon_input.text().strip(), "epsilon_outer")
            outer_max_iterations = int(parse_localized_float(self.outer_max_iterations_input.text().strip(), "max_outer_iterations"))
            inner_epsilon = parse_localized_float(self.inner_epsilon_input.text().strip(), "epsilon")
            inner_max_iterations = int(parse_localized_float(self.inner_max_iterations_input.text().strip(), "max_iterations"))
            if mu0 <= 0.0:
                raise ValueError("Начальное μ должно быть > 0.")
            if not (0.0 < beta < 1.0):
                raise ValueError("Коэффициент уменьшения β должен быть в диапазоне (0, 1).")
            if outer_epsilon <= 0.0:
                raise ValueError("Точность внешнего цикла должна быть > 0.")
            if outer_max_iterations < 1:
                raise ValueError("Макс. число внешних итераций должно быть >= 1.")
            if inner_epsilon <= 0.0:
                raise ValueError("epsilon внутренних итераций должен быть > 0.")
            if inner_max_iterations <= 0:
                raise ValueError("max_iterations внутренних итераций должен быть > 0.")
        except ValueError as exc:
            self._set_status(str(exc), kind="error")
            return

        if not is_strictly_feasible(start_point, self._problem.constraints):
            message = "Стартовая точка должна быть строго внутренней: g_i(x0) < 0 для всех ограничений."
            self._set_status(message, kind="error")
            return

        config = build_method_config(
            mu0=mu0,
            beta=beta,
            epsilon_outer=outer_epsilon,
            max_outer_iterations=outer_max_iterations,
            barrier_kind=self._selected_barrier_kind(),
            inner_epsilon=inner_epsilon,
            inner_max_iterations=inner_max_iterations,
        )

        self._set_running_state(True)
        self._set_status("Выполняются внутренние итерации Розенброка...", kind="info")
        self._run_task.start(
            "lr5_barrier_run",
            lambda: run_barrier_method(self._problem, start_point, config),
        )

    def _on_run_succeeded(self, result: BarrierResult) -> None:
        self._run_state = RunState(result=result, trace_id=uuid.uuid4().hex[:12])
        self._render_result(result)
        self._set_running_state(False)
        if result.status == "success":
            self._set_status("Расчёт завершён. Можно сохранить результаты.", kind="success")
        elif result.status == "warning":
            self._set_status("Расчёт остановлен с предупреждением. Последняя допустимая точка сохранена.", kind="warning")
        else:
            self._set_status("Расчёт завершён без валидного допустимого результата.", kind="error")

    def _on_run_failed(self, message: str, traceback_text: str) -> None:
        _ = traceback_text
        self._set_running_state(False)
        self.save_button.setEnabled(self._run_state is not None)
        self._set_status(message, kind="error")

    def _render_result(self, result: BarrierResult) -> None:
        self._render_summary(result)
        self._render_outer_table(result)
        self._render_plot(result)
        if self.report_empty_stack is not None:
            self.report_empty_stack.set_empty(False)
        self.plot_empty_stack.set_empty(False)
        if self.results_tabs is not None:
            self.results_tabs.setCurrentIndex(self.results_tab_indexes.results)
        if result.iterations:
            self.outer_table.selectRow(0)
            self._render_inner_table(result.iterations[0])
        elif result.failed_outer_iteration is not None:
            self._clear_inner_table("Внешний шаг нарушил допустимость. Валидной итерации для показа нет.")
        else:
            self._clear_inner_table("Нет внешних итераций для отображения.")

    def _render_summary(self, result: BarrierResult) -> None:
        self.summary_status.setText(self._summary_status_text(result))
        self.summary_comment.setText(self._summary_status_comment(result))
        self.summary_point_caption.setText(self._summary_point_caption(result))
        self.summary_barrier_caption.setText("Вид барьера")
        self.summary_mu_caption.setText("Последнее μ")
        self.summary_barrier.setText(self._barrier_label(result.config.barrier_kind))
        last_valid = result.last_valid_outer_iteration
        if last_valid is not None:
            self.summary_point.setText(self._format_point(last_valid.x_mu_k))
            self.summary_value.setText(self._format_float(last_valid.objective_value))
            self.summary_mu.setText(self._format_float(last_valid.mu_k))
            self.summary_constraints.setText(self._summary_constraints_text(last_valid.constraints_values))
        else:
            self.summary_point.setText("—")
            self.summary_value.setText("—")
            self.summary_mu.setText("—")
            self.summary_constraints.setText("g1(x) = —\ng2(x) = —")
        self.summary_conclusion.setText(self._human_result_status(result))

    def _render_outer_table(self, result: BarrierResult) -> None:
        iterations = result.iterations
        self.outer_table.setRowCount(len(iterations))
        for row_index, iteration in enumerate(iterations):
            values = (
                str(iteration.k),
                self._format_float(iteration.mu_k),
                self._format_point(iteration.x_mu_k),
                self._format_float(iteration.objective_value),
                self._format_float(iteration.barrier_metric),
                self._format_float(iteration.barrier_metric_term),
                self._format_float(iteration.theta_value),
                *[self._format_float(value) for value in iteration.constraints_values],
                str(iteration.inner_result.iterations_count),
            )
            for column_index, text in enumerate(values):
                item = QTableWidgetItem(text)
                numeric_columns = {0, 1, 3, 4, 5, 6, 7 + len(iteration.constraints_values)}
                numeric_columns.update(range(7, 7 + len(iteration.constraints_values)))
                if column_index in numeric_columns:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.outer_table.setItem(row_index, column_index, item)
        set_table_data_layout(
            self.outer_table,
            self._outer_table_widths(len(result.problem.constraints)),
        )

    def _render_inner_table(self, iteration: BarrierIterationResult) -> None:
        inner_result = iteration.inner_result
        self.inner_label.setText(
            f"Внешняя итерация k = {iteration.k}, μ = {iteration.mu_k:.10g}. "
            f"Внутренних шагов: {inner_result.iterations_count}. "
            f"Стоп: {html.escape(inner_result.stop_reason)}"
        )
        self.inner_table.setRowCount(len(inner_result.steps))
        for row_index, step in enumerate(inner_result.steps):
            values = (
                str(step.k),
                self._format_point(step.x_k),
                self._format_float(step.f_x_k),
                str(step.j),
                self._format_point(step.direction),
                self._format_point(step.y_j),
                self._format_float(step.f_y_j),
                self._format_float(step.lambda_j),
                self._format_point(step.y_next),
                self._format_float(step.f_y_next),
            )
            for column_index, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.inner_table.setItem(row_index, column_index, item)
        set_table_data_layout(self.inner_table, [48, 158, 118, 48, 150, 150, 118, 118, 150, 118])

    def _clear_inner_table(self, message: str) -> None:
        self.inner_table.setRowCount(0)
        self.inner_label.setText(message)

    def _on_outer_selection_changed(self) -> None:
        state = self._run_state
        if state is None:
            return
        row = self.outer_table.currentRow()
        if row < 0 or row >= len(state.result.iterations):
            return
        self._selected_iteration_index = row
        self._render_inner_table(state.result.iterations[row])

    def _render_plot(self, result: BarrierResult) -> None:
        points = self._collect_points(result)
        x_min, x_max, y_min, y_max = self._plot_bounds(points)
        grid_x = np.linspace(x_min, x_max, 240)
        grid_y = np.linspace(y_min, y_max, 240)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        objective_mesh = (mesh_x - 5.0) ** 2 + (mesh_y - 3.0) ** 2
        g1_mesh = -mesh_x + 2.0 * mesh_y - 4.0
        g2_mesh = mesh_x + mesh_y - 3.0
        feasible_mask = np.where((g1_mesh <= 0.0) & (g2_mesh <= 0.0), 1.0, 0.0)

        with dark_plot_context():
            figure = self.canvas.figure
            figure.clear()
            figure.patch.set_facecolor("#171b24")
            axis = figure.add_subplot(1, 1, 1)
            axis.set_facecolor("#10141f")
            axis.contourf(mesh_x, mesh_y, feasible_mask, levels=[-0.5, 0.5, 1.5], colors=["#1f7a3d"], alpha=0.16)
            contour = axis.contour(mesh_x, mesh_y, objective_mesh, levels=20, cmap="turbo", linewidths=0.9)
            axis.clabel(contour, inline=True, fontsize=8, colors="#dce6f5")
            axis.contour(mesh_x, mesh_y, g1_mesh, levels=[0.0], colors="#ff8c42", linewidths=2.1)
            axis.contour(mesh_x, mesh_y, g2_mesh, levels=[0.0], colors="#6ee7ff", linewidths=2.1)

            outer_points = [result.start_point, *[iteration.x_mu_k for iteration in result.iterations]]
            if outer_points:
                outer_x = [point[0] for point in outer_points]
                outer_y = [point[1] for point in outer_points]
                axis.plot(
                    outer_x,
                    outer_y,
                    color="#f0f3f8",
                    linewidth=2.6,
                    marker="o",
                    markersize=4.8,
                    zorder=4,
                )
                axis.scatter([outer_x[0]], [outer_y[0]], color="#2da3ff", s=100, label="Стартовая точка", zorder=5)
                if result.last_valid_outer_iteration is not None:
                    final_label = "Итоговая точка" if result.status == "success" else "Последняя допустимая точка"
                    axis.scatter([outer_x[-1]], [outer_y[-1]], color="#57d773", s=100, label=final_label, zorder=5)

            for index, iteration in enumerate(result.iterations):
                trajectory = list(iteration.inner_result.trajectory)
                if len(trajectory) < 2 or index > 4:
                    continue
                xs = [point[0] for point in trajectory]
                ys = [point[1] for point in trajectory]
                axis.plot(xs, ys, color="#ff4f87", alpha=0.24, linewidth=1.15, marker="o", markersize=2.8, zorder=3)

            axis.set_title("Траектория метода барьерных функций")
            axis.set_xlabel("x₁")
            axis.set_ylabel("x₂")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.14)
            legend_handles = [
                Patch(facecolor="#1f7a3d", edgecolor="none", alpha=0.18, label="Допустимая область"),
                Line2D([], [], color="#ff8c42", linewidth=2.1, label="Ограничение g1(x)=0"),
                Line2D([], [], color="#6ee7ff", linewidth=2.1, label="Ограничение g2(x)=0"),
                Line2D([], [], color="#f0f3f8", linewidth=2.6, marker="o", markersize=4.8, label="Внешняя траектория"),
                Line2D([], [], color="#ff4f87", linewidth=1.15, marker="o", markersize=2.8, alpha=0.24, label="Внутренние шаги"),
                Line2D([], [], color="none", marker="o", markerfacecolor="#2da3ff", markeredgecolor="#ffffff", markersize=8, label="Стартовая точка"),
                *(
                    [
                        Line2D(
                            [],
                            [],
                            color="none",
                            marker="o",
                            markerfacecolor="#57d773",
                            markeredgecolor="#ffffff",
                            markersize=8,
                            label="Итоговая точка" if result.status == "success" else "Последняя допустимая точка",
                        )
                    ]
                    if result.last_valid_outer_iteration is not None
                    else []
                ),
            ]
            axis.legend(handles=legend_handles, loc="best", framealpha=0.92, fontsize=9, ncol=2)
            self.canvas.draw_idle()

    def _collect_points(self, result: BarrierResult) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = [result.start_point, result.optimum_point]
        for iteration in result.iterations:
            points.extend(iteration.inner_result.trajectory)
            points.append(iteration.x_mu_k)
        return points

    def _plot_bounds(self, points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        return (
            x_min - x_span * 0.8,
            x_max + x_span * 0.8,
            y_min - y_span * 0.8,
            y_max + y_span * 0.8,
        )

    def _reset_inputs(self) -> None:
        start_x1, start_x2 = DEFAULT_START_POINT.split(";")
        self.start_x1_input.setText(start_x1)
        self.start_x2_input.setText(start_x2)
        self.mu0_input.setText(DEFAULT_MU0_TEXT)
        self.beta_input.setText(DEFAULT_BETA_TEXT)
        self.outer_epsilon_input.setText(DEFAULT_EPSILON_OUTER_TEXT)
        self.outer_max_iterations_input.setText(DEFAULT_MAX_OUTER_ITERATIONS_TEXT)
        self.inner_epsilon_input.setText(DEFAULT_INNER_EPSILON)
        self.inner_max_iterations_input.setText(DEFAULT_INNER_MAX_ITERATIONS)
        self.barrier_buttons["reciprocal"].setChecked(True)
        self._on_barrier_selected("reciprocal", True)
        self._reset_view()

    def _reset_view(self) -> None:
        self._run_state = None
        self._selected_iteration_index = None
        self.save_button.setEnabled(False)
        self.advanced_toggle.setChecked(False)
        self.inner_details_toggle.setChecked(False)
        self.summary_status.setText("Пока нет результатов")
        self.summary_comment.setText("Задай параметры слева и нажми «Рассчитать».")
        self.summary_point_caption.setText("Итоговая точка")
        self.summary_barrier_caption.setText("Вид барьера")
        self.summary_mu_caption.setText("Последнее μ")
        self.summary_point.setText("—")
        self.summary_value.setText("—")
        self.summary_barrier.setText("—")
        self.summary_mu.setText("—")
        self.summary_constraints.setText("g1(x) = —\ng2(x) = —")
        self.summary_conclusion.setText("Пока нет результата.")
        self._clear_inner_table("Выберите внешнюю итерацию, чтобы увидеть внутреннюю историю.")
        self.outer_table.setRowCount(0)
        clear_plot_canvas(self.canvas, "Пока нет результатов.\nНажми «Рассчитать», чтобы построить график.")
        if self.report_empty_stack is not None:
            self.report_empty_stack.set_empty(True)
        self.plot_empty_stack.set_empty(True)
        if self.results_tabs is not None:
            self.results_tabs.setCurrentIndex(0)
        self._set_status("Готов к запуску.", kind="info")

    def _save_result(self) -> None:
        state = self._run_state
        if state is None:
            message = "Нет результата для сохранения."
            self._set_status(message, kind="error")
            return
        run_dir = self._artifacts_store.save_result(state.result, state.trace_id)
        self._set_status(f"Результаты сохранены в {run_dir}", kind="success")

    def _set_running_state(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.reset_button.setEnabled(not running)
        self.save_button.setEnabled((not running) and self._run_state is not None)

    def _set_status(self, message: str, *, kind: str) -> None:
        prefix = {
            "info": "Статус",
            "success": "Готово",
            "warning": "Предупреждение",
            "error": "Ошибка",
        }.get(kind, "Статус")
        self.status_label.setText(f"<b>{prefix}:</b> {html.escape(message)}")

    @staticmethod
    def _format_float(value: float) -> str:
        if not math.isfinite(value):
            return "invalid"
        return f"{value:.10g}"

    @staticmethod
    def _format_point(point: tuple[float, float]) -> str:
        return f"({point[0]:.10g}; {point[1]:.10g})"

    @staticmethod
    def _build_summary_metric_card(title: str, value: str) -> tuple[QWidget, QLabel, QLabel]:
        card = QWidget()
        card.setObjectName("SummaryMetricCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(3)

        caption = QLabel(title)
        caption.setProperty("role", "summary-caption")
        caption.setWordWrap(True)
        caption.setTextFormat(Qt.PlainText)
        card_layout.addWidget(caption)

        value_label = QLabel(value)
        value_label.setProperty("role", "summary-value")
        value_label.setWordWrap(True)
        value_label.setTextFormat(Qt.PlainText)
        card_layout.addWidget(value_label)
        return card, caption, value_label

    def _build_collapsible_header(self, title: str) -> tuple[QWidget, QToolButton]:
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(2, 4, 0, 0)
        header_layout.setSpacing(8)

        toggle = QToolButton()
        toggle.setProperty("role", "section-toggle")
        toggle.setCheckable(True)
        toggle.setChecked(False)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.RightArrow)
        toggle.setText(title)
        toggle.setAutoRaise(True)
        toggle.setCursor(Qt.PointingHandCursor)
        toggle.toggled.connect(lambda checked: self._sync_collapsible_toggle(toggle, checked))
        header_layout.addWidget(toggle)
        header_layout.addStretch(1)
        return header, toggle

    @staticmethod
    def _bind_toggle(toggle: QToolButton, content: QWidget, *, expanded: bool) -> None:
        content.setVisible(expanded)
        toggle.setChecked(expanded)
        toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        toggle.toggled.connect(
            lambda checked: (
                content.setVisible(checked),
                toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow),
            )
        )

    def _sync_collapsible_toggle(self, toggle: QToolButton, checked: bool) -> None:
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    @staticmethod
    def _barrier_label(kind: str) -> str:
        return {
            "reciprocal": "Обратный барьер",
            "log": "Логарифмический барьер",
        }.get(kind, kind)

    @staticmethod
    def _summary_constraints_text(constraints: tuple[float, ...] | None) -> str:
        if not constraints:
            return "g1(x) = —\ng2(x) = —"
        values = [f"g{index + 1}(x) = {value:.10g}" for index, value in enumerate(constraints)]
        return "\n".join(values)

    def _summary_status_text(self, result: BarrierResult) -> str:
        if result.status == "success":
            return "Готово"
        if result.status == "warning":
            return "Расчёт остановлен с предупреждением"
        return "Ошибка"

    def _summary_status_comment(self, result: BarrierResult) -> str:
        if result.status == "success":
            return "Внешний цикл завершён на последней допустимой точке."
        return result.stop_reason

    @staticmethod
    def _outer_table_widths(constraint_count: int) -> list[int]:
        return [42, 82, 154, 102, 102, 112, 102, *([110] * constraint_count), 76]

    def _human_result_status(self, result: BarrierResult) -> str:
        if result.status == "success":
            return "Получено корректное допустимое решение."
        if result.status == "warning":
            if result.last_valid_outer_iteration is not None:
                return "Показана последняя допустимая точка; следующий шаг оказался недопустимым."
            return "Расчёт остановлен до получения допустимого решения."
        return "Валидное допустимое решение не найдено."

    @staticmethod
    def _summary_point_caption(result: BarrierResult) -> str:
        if result.status == "success":
            return "Итоговая точка"
        return "Последняя допустимая точка"


def main(argv: list[str] | None = None) -> None:
    _ = argv
    app = QApplication.instance() or QApplication(sys.argv)
    window = BarrierWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
