"""GUI для ЛР3: градиентные методы в стиле ЛР1/ЛР2."""

from __future__ import annotations

import html
import math
import sys
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from optim_core.ui import (
    ControlsPanel,
    DarkQtThemeTokens,
    PlotCanvas,
    TaskController,
    add_parameter_row,
    build_choice_chip_styles,
    build_dark_qt_base_styles,
    clear_plot_canvas,
    configure_two_panel_splitter,
    create_choice_chip_grid,
    create_controls_panel,
    configure_data_table,
    create_parameter_grid,
    create_primary_action_button,
    create_results_workspace,
    create_scroll_container,
    create_standard_group,
    dark_plot_context,
)
from optim_core.parsing import parse_localized_float
from PySide6.QtCore import Qt  # type: ignore[import-not-found]
from PySide6.QtWidgets import (  # type: ignore[import-not-found]
    QApplication,
    QButtonGroup,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from lr3.application.services import (
    ServiceMetrics,
    build_config,
    build_start_point,
    run_conjugate,
    run_gradient,
)
from lr3.domain.expression import analyze_local_extremum, compile_objective
from lr3.domain.models import ExtremumAnalysis, Goal, IterationRecord, MethodConfig, OptimizationResult, Point2D

APP_TITLE = "ЛР3 — Градиентные методы"

FUNCTION_TERMS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("1", (0, 0)),
    ("x₁", (1, 0)),
    ("x₂", (0, 1)),
    ("x₁²", (2, 0)),
    ("x₁x₂", (1, 1)),
    ("x₂²", (0, 2)),
)

FUNCTION_PRESET_CONFIGS = {
    "gradient": {
        "label": "Градиентная",
        "tooltip": "Функция из задания для метода первого порядка.",
        "coefficients": (0.0, 1.0, -2.0, 1.0, -1.0, 1.0),
        "method": "gradient",
        "goal": "min",
    },
    "conjugate": {
        "label": "Сопр. градиенты",
        "tooltip": "Функция из задания для метода сопряжённых градиентов.",
        "coefficients": (-2.0, -1.0, -2.0, -0.1, 0.0, -100.0),
        "method": "conjugate",
        "goal": "max",
    },
    "custom": {
        "label": "Пользовательская",
        "tooltip": "Своя функция. Коэффициенты можно редактировать вручную.",
        "coefficients": None,
        "method": None,
        "goal": None,
    },
}

GOAL_OPTIONS = (("Минимум", "min"), ("Максимум", "max"))

METHOD_DEFAULTS = {
    "gradient": {
        "x1": "0",
        "x2": "0",
        "step": "0.1",
        "eps": "1e-6",
        "iters": "1000",
    },
    "conjugate": {
        "x1": "1",
        "x2": "1",
        "step": "0.2",
        "eps": "1e-6",
        "iters": "300",
    },
}

DEFAULT_TIMEOUT_SECONDS = "3.0"
DEFAULT_MIN_STEP = "1e-8"
DEFAULT_GRADIENT_STEP = "1e-6"
DEFAULT_MAX_STEP_EXPANSIONS = "16"


@dataclass(frozen=True)
class RunPayload:
    """Результат одного запуска для UI-потока."""

    expression: str
    config: MethodConfig
    result: OptimizationResult
    metrics: ServiceMetrics


class GradientMethodsWindow(QMainWindow):
    """Главное окно ЛР3 в Qt-стеке проекта."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1530, 960)
        self._apply_styles()

        self._last_payload: RunPayload | None = None
        self._run_task = TaskController(self)
        self._run_task.succeeded.connect(self._on_run_succeeded)
        self._run_task.failed.connect(self._on_run_failed)
        self._active_preset_key = "gradient"
        self._custom_coefficients_cache = FUNCTION_PRESET_CONFIGS["gradient"]["coefficients"]
        self._updating_coefficients = False

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

        self.preset_buttons["gradient"].setChecked(True)
        self._apply_preset("gradient")
        self.method_buttons["gradient"].setChecked(True)
        self._set_method_defaults("gradient", True)
        self.goal_buttons["min"].setChecked(True)
        self._set_goal_defaults("min", True)
        self._set_results_empty_state(True)
        self._clear_plot()

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
                min-height: 23px;
                font-size: 12px;
                padding: 4px 8px;
            }
            QLabel#SectionCaption {
                font-size: 10px;
            }
            QLabel[role="parameter-label"] {
                font-size: 13px;
                font-weight: 600;
            }
            QLabel[role="formula-preview"] {
                min-height: 72px;
                text-align: center;
                padding: 10px 12px;
                font-family: "Cambria Math", "Times New Roman", serif;
                font-size: 15px;
                font-weight: 700;
                background: #101722;
                border: 1px solid #31425a;
                border-radius: 10px;
            }
            QLabel[role="report-caption"] {
                line-height: 1.2;
                font-size: 11px;
            }
            QLabel[role="report-value"] {
                font-size: 12px;
                line-height: 1.2;
            }
            QGroupBox#ReportCard,
            QGroupBox#IterationCard {
                padding-top: 12px;
            }
            QPushButton[role="choice-chip"] {
                min-height: 25px;
                padding: 4px 8px;
                border-radius: 8px;
                background: #243045;
                border: 1px solid #3e4c68;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton[role="choice-chip"]:checked {
                background: #1f63bf;
                border-color: #3474cc;
            }
            QTableWidget#CoefficientTable {
                background: #131824;
                border: 1px solid #3f4a62;
                border-radius: 8px;
                gridline-color: #2d3241;
                color: #f5f7fb;
                font-size: 12px;
            }
            QTableWidget#CoefficientTable::item {
                padding: 2px 4px;
            }
            QTableWidget#CoefficientTable QHeaderView::section {
                background: #222938;
                color: #dbe2ee;
                border-right: 1px solid #33415b;
                border-bottom: 1px solid #33415b;
                padding: 4px 6px;
                font-size: 10px;
                font-weight: 700;
            }
            QSplitter::handle {
                background: #2a3549;
                border-radius: 3px;
            }
            """
        )

    def _build_controls_panel(self) -> QWidget:
        controls: ControlsPanel = create_controls_panel(min_width=490, max_width=540, spacing=7)
        panel = controls.panel
        layout = controls.layout

        basic_group, basic_layout = create_standard_group("Основные", spacing=4, margins=(12, 12, 12, 10))
        preset_caption = QLabel("Пресет функции")
        preset_caption.setObjectName("SectionCaption")
        basic_layout.addWidget(preset_caption)

        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        preset_keys = ("gradient", "conjugate", "custom")
        preset_row, preset_buttons = create_choice_chip_grid(
            group=self.preset_group,
            options=tuple((FUNCTION_PRESET_CONFIGS[key]["label"], key) for key in preset_keys),
            columns=3,
            horizontal_spacing=3,
            vertical_spacing=3,
            on_clicked=self._on_preset_selected,
            tooltips={key: FUNCTION_PRESET_CONFIGS[key]["tooltip"] for key in preset_keys},
        )
        self.preset_buttons = {key: button for key, button in zip(preset_keys, preset_buttons, strict=True)}
        basic_layout.addWidget(preset_row)

        coeff_caption = QLabel("Коэффициенты функции")
        coeff_caption.setObjectName("SectionCaption")
        basic_layout.addWidget(coeff_caption)

        self.coefficient_table = QTableWidget(len(FUNCTION_TERMS), 2)
        self.coefficient_table.setObjectName("CoefficientTable")
        self.coefficient_table.setHorizontalHeaderLabels(["Член", "Коэффициент"])
        configure_data_table(
            self.coefficient_table,
            min_row_height=28,
            allow_selection=False,
            allow_editing=True,
            word_wrap=False,
        )
        self.coefficient_table.verticalHeader().setVisible(False)
        self.coefficient_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.coefficient_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.coefficient_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.coefficient_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.coefficient_table.setMinimumHeight(204)
        self.coefficient_table.setMaximumHeight(220)
        self.coefficient_table.itemChanged.connect(self._on_coefficient_item_changed)
        self._configure_coefficient_table()
        basic_layout.addWidget(self.coefficient_table)

        self.formula_preview = QLabel()
        self.formula_preview.setProperty("role", "formula-preview")
        self.formula_preview.setTextFormat(Qt.PlainText)
        self.formula_preview.setWordWrap(True)
        self.formula_preview.setAlignment(Qt.AlignCenter)
        self.formula_preview.setMinimumHeight(76)
        basic_layout.addWidget(self.formula_preview)

        method_caption = QLabel("Метод")
        method_caption.setObjectName("SectionCaption")
        basic_layout.addWidget(method_caption)

        self.method_group = QButtonGroup(self)
        self.method_group.setExclusive(True)
        method_keys = ("gradient", "conjugate")
        method_row, method_buttons = create_choice_chip_grid(
            group=self.method_group,
            options=(("Градиентный", "gradient"), ("Сопр. градиенты", "conjugate")),
            columns=2,
            horizontal_spacing=3,
            vertical_spacing=3,
            on_clicked=self._set_method_defaults,
        )
        self.method_buttons = {key: button for key, button in zip(method_keys, method_buttons, strict=True)}
        basic_layout.addWidget(method_row)

        goal_caption = QLabel("Цель")
        goal_caption.setObjectName("SectionCaption")
        basic_layout.addWidget(goal_caption)

        self.goal_group = QButtonGroup(self)
        self.goal_group.setExclusive(True)
        goal_row, goal_buttons = create_choice_chip_grid(
            group=self.goal_group,
            options=GOAL_OPTIONS,
            columns=2,
            horizontal_spacing=3,
            vertical_spacing=3,
            on_clicked=self._set_goal_defaults,
        )
        self.goal_buttons = {key: button for key, button in zip(("min", "max"), goal_buttons, strict=True)}
        basic_layout.addWidget(goal_row)

        self.start_x1_input = QLineEdit()
        self.start_x2_input = QLineEdit()
        self.epsilon_input = QLineEdit()
        self.initial_step_input = QLineEdit()

        start_caption = QLabel("Стартовая точка")
        start_caption.setObjectName("SectionCaption")
        basic_layout.addWidget(start_caption)

        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(6)
        start_x1_caption = QLabel("x₁")
        start_x1_caption.setObjectName("SectionCaption")
        start_x2_caption = QLabel("x₂")
        start_x2_caption.setObjectName("SectionCaption")
        self.start_x1_input.setMinimumWidth(88)
        self.start_x2_input.setMinimumWidth(88)
        start_layout.addWidget(start_x1_caption)
        start_layout.addWidget(self.start_x1_input, 1)
        start_layout.addWidget(start_x2_caption)
        start_layout.addWidget(self.start_x2_input, 1)

        basic_layout.addWidget(start_row)

        advanced_group = QGroupBox("Дополнительные")
        advanced_layout = create_parameter_grid(advanced_group)

        add_parameter_row(advanced_layout, row=0, label="Точность ε", control=self.epsilon_input)
        add_parameter_row(advanced_layout, row=1, label="Начальный шаг", control=self.initial_step_input)

        self.run_button = create_primary_action_button(text="Рассчитать", on_click=self._run_clicked)

        layout.addWidget(basic_group)
        layout.addWidget(advanced_group)
        layout.addWidget(self.run_button)
        layout.addStretch(1)
        return panel

    def _build_results_panel(self) -> QWidget:
        workspace = create_results_workspace(
            results_title="Отчёт",
            plot_title="Графики",
            with_tables_empty_state=True,
            tables_empty_title="Пока нет результатов",
            tables_empty_description=(
                "Слева выбери метод, формулу и параметры расчёта.\n"
                "После запуска здесь появится учебный отчёт по итерациям и графики."
            ),
            tables_empty_hint="Нажми «Рассчитать», чтобы получить отчёт и графики.",
        )

        self.results_empty_stack = workspace.tables_empty_stack
        if self.results_empty_stack is None:
            raise RuntimeError("Ожидался EmptyStateStack для отчётной вкладки")

        report_group = QGroupBox("Учебный отчёт")
        report_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        report_layout = QVBoxLayout(report_group)
        report_layout.setContentsMargins(14, 12, 14, 14)
        report_layout.setSpacing(10)

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
        report_layout.addWidget(self.report_scroll)
        workspace.tables_layout.addWidget(report_group)

        plot_group = QGroupBox("Графики")
        plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout = QVBoxLayout(plot_group)

        self.plot_context_label = QLabel("")
        self.plot_context_label.setObjectName("SectionHint")
        self.plot_context_label.setWordWrap(True)
        self.plot_context_label.hide()
        plot_layout.addWidget(self.plot_context_label)

        self.plot_state_label = QLabel("График появится после расчёта.")
        self.plot_state_label.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.plot_state_label)

        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setFrameShape(QScrollArea.NoFrame)
        plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_scroll.verticalScrollBar().setSingleStep(32)

        plot_host = QWidget()
        plot_host_layout = QHBoxLayout(plot_host)
        plot_host_layout.setContentsMargins(0, 0, 0, 0)
        plot_host_layout.setSpacing(0)

        self.canvas = PlotCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(620)
        plot_host_layout.addWidget(self.canvas)
        plot_scroll.setWidget(plot_host)
        plot_layout.addWidget(plot_scroll)
        workspace.plots_layout.addWidget(plot_group)

        return workspace.panel

    def _on_preset_selected(self, preset_key: str, checked: bool = True) -> None:
        if not checked:
            return
        if self._active_preset_key == "custom" and preset_key != "custom":
            self._custom_coefficients_cache = self._read_coefficients()
        self._active_preset_key = preset_key
        self._invalidate_current_result()
        for key, button in self.preset_buttons.items():
            button.setChecked(key == preset_key)
        self._apply_preset(preset_key)

    def _apply_preset(self, preset_key: str) -> None:
        preset = FUNCTION_PRESET_CONFIGS[preset_key]
        coefficients = preset["coefficients"]
        if coefficients is None:
            coefficients = self._custom_coefficients_cache
        self._set_coefficients(coefficients)
        self.coefficient_table.setEnabled(preset_key == "custom")
        self._update_formula_preview()
        if preset["method"] is not None:
            self.method_buttons[preset["method"]].setChecked(True)
            self._set_method_defaults(preset["method"], True)
        if preset["goal"] is not None:
            self.goal_buttons[preset["goal"]].setChecked(True)
            self._set_goal_defaults(preset["goal"], True)

    def _set_method_defaults(self, method: str, checked: bool = True) -> None:
        if not checked:
            return
        config = METHOD_DEFAULTS[method]
        self.start_x1_input.setText(config["x1"])
        self.start_x2_input.setText(config["x2"])
        self.initial_step_input.setText(config["step"])
        self.epsilon_input.setText(config["eps"])
        self._update_formula_preview()

    def _set_goal_defaults(self, goal: str, checked: bool = True) -> None:
        if not checked:
            return
        self._invalidate_current_result()
        self._update_formula_preview()

    def _run_clicked(self) -> None:
        if self._run_task.is_running():
            return
        method = self._selected_method()
        if method is None:
            QMessageBox.critical(self, "Ошибка ввода", "Выбери метод оптимизации.")
            return
        expression = self._current_expression()

        self._set_busy(True)
        self._run_task.start(
            "lr3-run",
            lambda: self._run_method(
                method=method,
                expression=expression,
                x1_raw=self.start_x1_input.text(),
                x2_raw=self.start_x2_input.text(),
                epsilon_raw=self.epsilon_input.text(),
                max_iterations_raw=self._selected_method_iterations(),
                initial_step_raw=self.initial_step_input.text(),
                timeout_raw=DEFAULT_TIMEOUT_SECONDS,
                goal_raw=self._selected_goal(),
                min_step_raw=DEFAULT_MIN_STEP,
                gradient_step_raw=DEFAULT_GRADIENT_STEP,
                max_step_expansions_raw=DEFAULT_MAX_STEP_EXPANSIONS,
            ),
        )

    def _run_method(
        self,
        *,
        method: str,
        expression: str,
        x1_raw: str,
        x2_raw: str,
        epsilon_raw: str,
        max_iterations_raw: str,
        initial_step_raw: str,
        timeout_raw: str,
        goal_raw: str,
        min_step_raw: str,
        gradient_step_raw: str,
        max_step_expansions_raw: str,
    ) -> RunPayload:
        if not expression:
            raise ValueError("Поле функции не должно быть пустым")

        start_point = build_start_point(x1_raw, x2_raw)
        config = build_config(
            epsilon_raw=epsilon_raw,
            max_iterations_raw=max_iterations_raw,
            initial_step_raw=initial_step_raw,
            timeout_raw=timeout_raw,
            goal_raw=goal_raw,
            min_step_raw=min_step_raw,
            gradient_step_raw=gradient_step_raw,
            max_step_expansions_raw=max_step_expansions_raw,
        )

        if method == "gradient":
            result, metrics = run_gradient(expression=expression, start_point=start_point, config=config)
        elif method == "conjugate":
            result, metrics = run_conjugate(expression=expression, start_point=start_point, config=config)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        return RunPayload(expression=expression, config=config, result=result, metrics=metrics)

    def _on_run_succeeded(self, payload: object) -> None:
        self._set_busy(False)
        if not isinstance(payload, RunPayload):
            QMessageBox.critical(self, "Ошибка расчёта", "Некорректный формат ответа вычислений.")
            return

        self._last_payload = payload
        self._render_report(payload)
        self._render_plot(payload)
        self._set_results_empty_state(False)

    def _on_run_failed(self, message: str, _stack: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Ошибка расчёта", message)

    def _render_report(self, payload: RunPayload) -> None:
        self._clear_report()

        result = payload.result
        objective = compile_objective(payload.expression)
        analysis = analyze_local_extremum(payload.expression, result.start_point, goal=payload.config.goal)

        self.report_layout.addWidget(
            self._build_task_card(
                payload=payload,
            )
        )
        self.report_layout.addWidget(
            self._build_analysis_card(
                analysis=analysis,
            )
        )

        iteration_cards = self._build_iteration_cards(payload, objective)
        if iteration_cards:
            for card in iteration_cards:
                self.report_layout.addWidget(card)
        else:
            self.report_layout.addWidget(self._build_empty_iterations_card())

        self.report_layout.addWidget(self._build_summary_card(payload, analysis))
        self.report_layout.addStretch(1)

    def _build_task_card(self, payload: RunPayload) -> QWidget:
        result = payload.result
        card, form = self._create_report_card("Постановка задачи")
        method_title = self._format_method_title(result.method_name)
        self._add_report_row(form, "Метод", self._math_text(method_title))
        self._add_report_row(form, "Цель", self._math_text(self._format_goal_task(payload.config.goal)))
        self._add_report_row(form, "Функция", self._math_text(f"F(x<sub>1</sub>, x<sub>2</sub>) = {self._format_formula(payload.expression)}"))
        self._add_report_row(form, "M<sub>0</sub>", self._math_text(self._format_point(result.start_point)))
        return card

    def _build_analysis_card(
        self,
        *,
        analysis: ExtremumAnalysis,
    ) -> QWidget:
        card, form = self._create_report_card("Аналитическая подготовка")
        self._add_report_row(
            form,
            "∇f(x<sub>1</sub>, x<sub>2</sub>)",
            self._math_text(
                self._format_expression_block(
                    "",
                    (
                        f"∂f/∂x<sub>1</sub> = {self._format_formula(analysis.gradient_formula[0])}",
                        f"∂f/∂x<sub>2</sub> = {self._format_formula(analysis.gradient_formula[1])}",
                    ),
                )
            ),
        )
        self._add_report_row(
            form,
            "∇f(M<sub>0</sub>)",
            self._math_text(
                self._format_vector_block(
                    analysis.gradient_at_start,
                    scalar_formatter=self._format_scalar,
                )
            ),
        )
        if analysis.stationary_points:
            self._add_report_row(
                form,
                "M*",
                self._math_text(self._format_stationary_points(analysis.stationary_points)),
            )
        else:
            self._add_report_row(form, "M*", self._math_text("не выделена"))
        self._add_report_row(
            form,
            "H",
            self._math_text(self._format_matrix_block(analysis.hessian_formula)),
        )
        self._add_report_row(form, "Вывод", self._math_text(self._build_analysis_conclusion(analysis)))
        return card

    def _build_iteration_cards(self, payload: RunPayload, objective: Callable[[Point2D], float]) -> list[QWidget]:
        result = payload.result
        if result.method_name == "gradient_ascent":
            return self._build_gradient_iteration_cards(payload, objective)
        if result.method_name == "conjugate_gradient_ascent":
            return self._build_conjugate_iteration_cards(payload)
        return []

    def _build_gradient_iteration_cards(self, payload: RunPayload, objective: Callable[[Point2D], float]) -> list[QWidget]:
        result = payload.result
        cards: list[QWidget] = []
        direction_multiplier = -1.0 if payload.config.goal == "min" else 1.0
        for record in result.records:
            has_step = record.step_size > 0.0
            step_direction = (
                (direction_multiplier * record.gradient[0], direction_multiplier * record.gradient[1])
                if has_step
                else None
            )
            new_point = self._translate_point(record.point, step_direction, record.step_size) if step_direction is not None else None
            new_value = objective(new_point) if new_point is not None else None
            note = self._gradient_step_note(record.step_size, payload.config.initial_step, result)
            card, form = self._create_report_card(f"Итерация {record.k + 1}", object_name="IterationCard")
            self._add_report_row(form, "Текущая точка", self._math_text(self._format_point(record.point)))
            self._add_report_row(form, "F(M<sub>k</sub>)", self._math_text(self._format_scalar(record.value)))
            self._add_report_row(
                form,
                "∇F(M<sub>k</sub>)",
                self._math_text(self._format_vector_block(record.gradient, scalar_formatter=self._format_scalar)),
            )
            self._add_report_row(form, "h<sub>k</sub>", self._math_text(self._format_step(record.step_size)))
            if new_point is not None and new_value is not None:
                operator = "+" if payload.config.goal == "max" else "-"
                self._add_report_row(
                    form,
                    f"M<sub>{record.k + 1}</sub>",
                    self._math_text(
                        f"M<sub>{record.k + 1}</sub> = M<sub>{record.k}</sub> {operator} h<sub>{record.k}</sub>·∇f(M<sub>{record.k}</sub>) = {self._format_point(new_point)}"
                    ),
                )
                self._add_report_row(
                    form,
                    f"F(M<sub>{record.k + 1}</sub>)",
                    self._math_text(self._format_scalar(new_value)),
                )
            else:
                self._add_report_row(form, "Переход", self._math_text("Шаг не выполнен"))
            self._add_report_row(form, "Комментарий", self._math_text(note))
            cards.append(card)
        return cards

    def _build_conjugate_iteration_cards(self, payload: RunPayload) -> list[QWidget]:
        result = payload.result
        cards: list[QWidget] = []
        for index, record in enumerate(result.records):
            note = self._conjugate_step_note(record, result)
            card, form = self._create_report_card(f"Итерация {index + 1}", object_name="IterationCard")
            self._add_report_row(form, "Текущая точка", self._math_text(self._format_point(record.point)))
            self._add_report_row(form, "F(M<sub>k</sub>)", self._math_text(self._format_scalar(record.value)))
            self._add_report_row(
                form,
                "∇F(M<sub>k</sub>)",
                self._math_text(self._format_vector_block(record.gradient, scalar_formatter=self._format_scalar)),
            )
            if record.direction is not None:
                self._add_report_row(
                    form,
                    "s<sub>k</sub>",
                    self._math_text(self._format_vector_block(record.direction, scalar_formatter=self._format_scalar)),
                )
            if record.beta is not None:
                self._add_report_row(form, "β<sub>k</sub>", self._math_text(self._format_scalar(record.beta)))
            self._add_report_row(form, "λ<sub>k</sub>", self._math_text(self._format_step(record.step_size)))
            if record.next_point is not None and record.next_value is not None:
                self._add_report_row(
                    form,
                    f"M<sub>{index + 1}</sub>",
                    self._math_text(
                        f"M<sub>{index + 1}</sub> = M<sub>{index}</sub> + λ<sub>{index}</sub>·s<sub>{index}</sub> = {self._format_point(record.next_point)}"
                    ),
                )
                self._add_report_row(
                    form,
                    f"F(M<sub>{index + 1}</sub>)",
                    self._math_text(self._format_scalar(record.next_value)),
                )
            else:
                self._add_report_row(form, "Переход", self._math_text("Шаг не выполнен"))
            self._add_report_row(form, "Комментарий", self._math_text(note))
            cards.append(card)
        return cards

    def _build_empty_iterations_card(self) -> QWidget:
        card, form = self._create_report_card("Итерации")
        self._add_report_row(form, "Статус", self._math_text("Итерации не выполнены"))
        return card

    def _build_summary_card(self, payload: RunPayload, analysis: ExtremumAnalysis) -> QWidget:
        result = payload.result
        card, form = self._create_report_card("Итог")
        self._add_report_row(form, "Нужно было найти", self._math_text(self._format_goal_task(payload.config.goal)))
        self._add_report_row(form, "Аналитика", self._math_text(analysis.stationary_point_kind))
        self._add_report_row(form, "Согласование", self._math_text(analysis.goal_alignment))
        self._add_report_row(form, "Численный метод", self._math_text(self._format_method_goal_conclusion(payload.config.goal, result.success, analysis.goal_alignment)))
        self._add_report_row(form, "Статус метода", self._math_text(self._format_success_text(result.success)))
        self._add_report_row(form, "Найденная точка", self._math_text(self._format_point(result.optimum_point)))
        self._add_report_row(form, "F(x*)", self._math_text(self._format_scalar(result.optimum_value)))
        self._add_report_row(form, "Итераций", self._math_text(str(result.iterations_count)))
        self._add_report_row(form, "Причина остановки", self._math_text(self._format_stop_reason(result.stop_reason)))

        if result.records:
            last_record = result.records[-1]
            last_norm = math.hypot(last_record.gradient[0], last_record.gradient[1])
            self._add_report_row(form, "Норма градиента", self._math_text(self._format_scalar(last_norm)))
            if last_record.step_size > 0.0:
                self._add_report_row(form, "Последний шаг", self._math_text(self._format_step(last_record.step_size)))
        return card

    def _clear_report(self) -> None:
        self._clear_layout(self.report_layout)

    def _create_report_card(self, title: str, *, object_name: str = "ReportCard") -> tuple[QGroupBox, QFormLayout]:
        card = QGroupBox(title)
        card.setObjectName(object_name)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card_layout = QFormLayout(card)
        card_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        card_layout.setFormAlignment(Qt.AlignTop)
        card_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        card_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        card_layout.setHorizontalSpacing(16)
        card_layout.setVerticalSpacing(8)
        card_layout.setContentsMargins(16, 14, 16, 14)
        return card, card_layout

    def _add_report_row(self, form: QFormLayout, caption: str, value: str) -> None:
        caption_label = QLabel(caption)
        caption_label.setProperty("role", "report-caption")
        caption_label.setTextFormat(Qt.RichText)
        caption_label.setWordWrap(True)

        value_label = QLabel(value)
        value_label.setProperty("role", "report-value")
        value_label.setTextFormat(Qt.RichText)
        value_label.setWordWrap(True)
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        form.addRow(caption_label, value_label)

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _translate_point(point: Point2D, direction: Point2D, scale: float) -> Point2D:
        return (point[0] + scale * direction[0], point[1] + scale * direction[1])

    def _gradient_step_note(self, step_size: float, initial_step: float, result: OptimizationResult) -> str:
        if step_size <= 0.0:
            if result.stop_reason == "gradient_norm_reached":
                return "Достигнута требуемая точность, переход не выполнялся."
            return "Улучшающий шаг не найден, метод остановлен."
        if math.isclose(step_size, initial_step, rel_tol=1e-9, abs_tol=1e-12):
            return f"Шаг принят без изменения: h = {self._format_step(step_size)}."
        if step_size < initial_step:
            return f"Шаг уменьшен до h = {self._format_step(step_size)}."
        return f"Шаг увеличен до h = {self._format_step(step_size)}."

    def _conjugate_step_note(self, record: IterationRecord, result: OptimizationResult) -> str:
        if record.step_size <= 0.0:
            if result.stop_reason == "gradient_norm_reached":
                return "Достигнута требуемая точность, переход не выполнялся."
            return "Улучшающий шаг не найден, метод остановлен."
        parts: list[str] = [f"Найден шаг λ = {self._format_step(record.step_size)}."]
        if record.beta is not None:
            parts.append(f"β = {self._format_scalar(record.beta)}.")
        if record.restart_direction:
            parts.append("Выполнен перезапуск направления.")
        return " ".join(parts)

    @staticmethod
    def _format_method_title(method_name: str) -> str:
        labels = {
            "gradient_ascent": "Градиентный метод первого порядка",
            "conjugate_gradient_ascent": "Метод сопряжённых градиентов Флетчера-Ривса",
        }
        return labels.get(method_name, method_name)

    @staticmethod
    def _format_goal_task(goal: Goal) -> str:
        return "Поиск минимума" if goal == "min" else "Поиск максимума"

    @staticmethod
    def _format_goal_target(goal: Goal) -> str:
        return "минимума" if goal == "min" else "максимума"

    @staticmethod
    def _format_goal_direction(goal: Goal) -> str:
        return "к минимуму" if goal == "min" else "к максимуму"

    @staticmethod
    def _format_method_goal_conclusion(goal: Goal, success: bool, goal_alignment: str) -> str:
        if success and "не согласуется" not in goal_alignment:
            return f"сходится {GradientMethodsWindow._format_goal_direction(goal)}"
        if success:
            return f"сошёлся, но цель {GradientMethodsWindow._format_goal_target(goal)} аналитикой не подтверждена"
        return "остановлен без подтверждения искомого экстремума"

    @staticmethod
    def _format_success_text(success: bool) -> str:
        return "метод завершился успешно" if success else "расчёт остановлен"

    @staticmethod
    def _format_point(point: Point2D) -> str:
        return f"({point[0]:.8f}; {point[1]:.8f})"

    @staticmethod
    def _format_scalar(value: float) -> str:
        return f"{value:.8g}"

    @staticmethod
    def _format_step(value: float) -> str:
        return f"{value:.8g}"

    @staticmethod
    def _format_formula(expression: str) -> str:
        readable = expression.replace("**", "^").replace("*", "·")
        readable = html.escape(readable)
        readable = readable.replace("x1", "x<sub>1</sub>").replace("x2", "x<sub>2</sub>")
        return readable

    @staticmethod
    def _format_stationary_points(points: tuple[Point2D, ...]) -> str:
        if not points:
            return "не найдены"
        if len(points) == 1:
            return f"M<sub>*</sub> = {GradientMethodsWindow._format_point(points[0])}"
        formatted = ", ".join(
            f"M<sub>*{index + 1}</sub> = {GradientMethodsWindow._format_point(point)}" for index, point in enumerate(points)
        )
        return formatted

    @staticmethod
    def _math_text(text: str) -> str:
        return (
            "<div style=\"font-family: 'Cambria Math', 'Times New Roman', serif; "
            "font-size: 12px; line-height: 1.18; font-weight: 500; white-space: normal;\">"
            f"{text}"
            "</div>"
        )

    @staticmethod
    def _format_expression_block(title: str, expressions: tuple[str, ...]) -> str:
        body = "".join(
            f"<div style=\"padding-left: 12px;\">{expression}</div>" for expression in expressions
        )
        return f"<div style=\"line-height: 1.35;\">{title}{body}</div>"

    @staticmethod
    def _format_vector_block(
        vector: tuple[object, ...],
        *,
        scalar_formatter: Callable[[float], str] | None = None,
    ) -> str:
        formatter = scalar_formatter or GradientMethodsWindow._format_scalar
        values = ", ".join(GradientMethodsWindow._format_math_value(value, formatter) for value in vector)
        return f"({values})"

    @staticmethod
    def _format_matrix_block(
        matrix: tuple[tuple[object, ...], ...],
        *,
        scalar_formatter: Callable[[float], str] | None = None,
    ) -> str:
        formatter = scalar_formatter or GradientMethodsWindow._format_scalar
        rows = [
            "<tr>"
            + "".join(
                f"<td style=\"padding: 0 8px; text-align: center;\">"
                f"{GradientMethodsWindow._format_math_value(value, formatter)}</td>"
                for value in row
            )
            + "</tr>"
            for row in matrix
        ]
        return (
            "<table style=\"border-collapse: collapse; margin: 0;\">"
            + "".join(rows)
            + "</table>"
        )

    @staticmethod
    def _format_math_value(value: object, numeric_formatter: Callable[[float], str]) -> str:
        if isinstance(value, str):
            return GradientMethodsWindow._format_formula(value)
        return numeric_formatter(float(value))

    @staticmethod
    def _build_analysis_conclusion(analysis: ExtremumAnalysis) -> str:
        parts: list[str] = ["<div>Вывод:</div>"]
        if analysis.stationary_points:
            marker = GradientMethodsWindow._format_stationary_points(analysis.stationary_points)
            if analysis.stationary_point_kind == "локальный минимум":
                parts.append(f"<div>Локальный минимум: {marker}.</div>")
            elif analysis.stationary_point_kind == "локальный максимум":
                parts.append(f"<div>Локальный максимум: {marker}.</div>")
            elif analysis.stationary_point_kind == "седловая точка":
                parts.append(f"<div>Седловая точка: {marker}.</div>")
            else:
                parts.append(f"<div>Искомый экстремум: {marker}.</div>")
        else:
            parts.append("<div>Искомый экстремум для выбранной цели отсутствует.</div>")

        parts.append(f"<div>Классификация: {html.escape(analysis.stationary_point_kind)}</div>")
        parts.append(f"<div>Согласование: {html.escape(analysis.goal_alignment)}</div>")
        parts.append(f"<div>{html.escape(analysis.theory_conclusion)}</div>")
        parts.append(f"<div>{html.escape(analysis.strictness_note)}</div>")
        return "".join(parts)

    def _configure_coefficient_table(self) -> None:
        self.coefficient_table.blockSignals(True)
        try:
            for row, (label, _) in enumerate(FUNCTION_TERMS):
                term_item = QTableWidgetItem(label)
                term_item.setFlags(Qt.ItemIsEnabled)
                term_item.setTextAlignment(Qt.AlignCenter)
                self.coefficient_table.setItem(row, 0, term_item)

                cell = QTableWidgetItem("0")
                cell.setTextAlignment(Qt.AlignCenter)
                self.coefficient_table.setItem(row, 1, cell)
        finally:
            self.coefficient_table.blockSignals(False)

    def _set_coefficients(self, coefficients: tuple[float, ...]) -> None:
        self._updating_coefficients = True
        self.coefficient_table.blockSignals(True)
        try:
            for row, value in enumerate(coefficients):
                item = self.coefficient_table.item(row, 1)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignCenter)
                    self.coefficient_table.setItem(row, 1, item)
                item.setText(self._format_scalar(value))
                item.setTextAlignment(Qt.AlignCenter)
        finally:
            self.coefficient_table.blockSignals(False)
            self._updating_coefficients = False

    def _read_coefficients(self) -> tuple[float, ...]:
        coefficients: list[float] = []
        for row in range(self.coefficient_table.rowCount()):
            item = self.coefficient_table.item(row, 1)
            raw = item.text().strip() if item and item.text().strip() else "0"
            coefficients.append(parse_localized_float(raw, f"c[{row}]"))
        return tuple(coefficients)

    def _on_coefficient_item_changed(self, _item: QTableWidgetItem) -> None:
        if self._updating_coefficients or _item.column() != 1:
            return
        if self._active_preset_key == "custom":
            self._custom_coefficients_cache = self._read_coefficients()
        self._update_formula_preview()
        self._invalidate_current_result()

    def _build_expression_from_coefficients(self, coefficients: tuple[float, ...]) -> str:
        terms: list[str] = []
        for value, (_, powers) in zip(coefficients, FUNCTION_TERMS, strict=True):
            if abs(value) < 1e-14:
                continue
            terms.append(self._build_expression_term(value, powers))
        return " + ".join(terms).replace("+ -", "- ") if terms else "0"

    def _build_expression_term(self, value: float, powers: tuple[int, int]) -> str:
        sign = "-" if value < 0 else ""
        abs_value = abs(value)
        coeff_text = self._format_scalar(abs_value)
        if powers == (0, 0):
            return f"{sign}{coeff_text}"
        factors: list[str] = []
        if abs_value != 1.0:
            factors.append(coeff_text)
        if powers[0] > 0:
            factors.append("x1" if powers[0] == 1 else f"x1**{powers[0]}")
        if powers[1] > 0:
            factors.append("x2" if powers[1] == 1 else f"x2**{powers[1]}")
        return f"{sign}{'*'.join(factors)}"

    def _format_polynomial_display(self, coefficients: tuple[float, ...]) -> str:
        terms: list[str] = []
        for value, (label, _) in zip(coefficients, FUNCTION_TERMS, strict=True):
            if abs(value) < 1e-14:
                continue
            terms.append(self._format_display_term(value, label))
        return " + ".join(terms).replace("+ -", "- ") if terms else "0"

    @staticmethod
    def _format_display_term(value: float, label: str) -> str:
        abs_value = abs(value)
        coeff_text = GradientMethodsWindow._format_scalar(abs_value)
        if label == "1":
            return f"-{coeff_text}" if value < 0 else coeff_text
        if abs_value == 1.0:
            return f"-{label}" if value < 0 else label
        return f"-{coeff_text}{label}" if value < 0 else f"{coeff_text}{label}"

    def _build_formula_preview_text(self, coefficients: tuple[float, ...]) -> str:
        display = self._format_polynomial_display(coefficients)
        return f"F(x₁, x₂) = {display}"

    def _current_expression(self) -> str:
        return self._build_expression_from_coefficients(self._read_coefficients())

    def _invalidate_current_result(self) -> None:
        self._last_payload = None
        self._clear_report()
        self._clear_plot()
        self._set_results_empty_state(True)

    def _update_formula_preview(self) -> None:
        coefficients = self._read_coefficients()
        if self._active_preset_key == "custom" and not self._updating_coefficients:
            self._custom_coefficients_cache = coefficients
        self.formula_preview.setText(self._build_formula_preview_text(coefficients))

    def _render_plot(self, payload: RunPayload) -> None:
        objective = compile_objective(payload.expression)
        result = payload.result
        goal = payload.config.goal
        goal_label = self._format_goal_target(goal)
        analysis = analyze_local_extremum(payload.expression, result.start_point, goal=goal)

        self.plot_state_label.hide()
        self.plot_context_label.setText(
            f"Цель: поиск {goal_label} | Метод: {self._format_method_title(result.method_name)} | "
            f"Итераций: {result.iterations_count} | Причина завершения: {self._format_stop_reason(result.stop_reason)} | "
            f"Аналитика: {analysis.goal_alignment}"
        )
        self.plot_context_label.show()

        with dark_plot_context():
            figure = self.canvas.figure
            figure.clear()
            figure.patch.set_facecolor("#171b24")

            ax_contour = figure.add_subplot(121)
            ax_convergence = figure.add_subplot(122)

            xs = [item.point[0] for item in result.records if math.isfinite(item.point[0]) and math.isfinite(item.point[1])]
            ys = [item.point[1] for item in result.records if math.isfinite(item.point[0]) and math.isfinite(item.point[1])]
            if not xs:
                xs = [result.start_point[0], result.optimum_point[0]]
                ys = [result.start_point[1], result.optimum_point[1]]

            x_min, x_max, y_min, y_max = self._build_plot_window(
                result=result,
                analysis=analysis,
                fallback_points=tuple(zip(xs, ys, strict=True)) if xs else (),
            )

            grid_x = np.linspace(x_min, x_max, 80)
            grid_y = np.linspace(y_min, y_max, 80)
            x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)
            z_mesh = np.full_like(x_mesh, np.nan, dtype=float)

            for i in range(x_mesh.shape[0]):
                for j in range(x_mesh.shape[1]):
                    try:
                        value = objective((float(x_mesh[i, j]), float(y_mesh[i, j])))
                    except OverflowError:
                        continue
                    if math.isfinite(value):
                        z_mesh[i, j] = value

            finite_z = z_mesh[np.isfinite(z_mesh)]
            if finite_z.size >= 2:
                low = float(np.percentile(finite_z, 5))
                high = float(np.percentile(finite_z, 95))
                if math.isfinite(low) and math.isfinite(high) and high > low:
                    contour = ax_contour.contourf(x_mesh, y_mesh, np.ma.masked_invalid(z_mesh), levels=25, cmap="viridis")
                    figure.colorbar(contour, ax=ax_contour)
                else:
                    ax_contour.text(0.5, 0.5, "График недоступен для выбранного масштаба", ha="center", va="center", transform=ax_contour.transAxes)
            else:
                ax_contour.text(0.5, 0.5, "График недоступен для выбранного масштаба", ha="center", va="center", transform=ax_contour.transAxes)

            if xs and ys:
                ax_contour.plot(xs, ys, "r.-", linewidth=2, markersize=6)
                ax_contour.scatter([xs[0]], [ys[0]], c="white", edgecolors="black", label="start")
            if math.isfinite(result.optimum_point[0]) and math.isfinite(result.optimum_point[1]):
                ax_contour.scatter(
                    [result.optimum_point[0]],
                    [result.optimum_point[1]],
                    c="#ff8c42" if goal == "max" else "#4fc3f7",
                    label=f"итог ({goal_label})",
                )
            if analysis.stationary_points:
                theoretical = analysis.stationary_points[0]
                ax_contour.scatter(
                    [theoretical[0]],
                    [theoretical[1]],
                    c="white",
                    edgecolors="black",
                    marker="*",
                    s=160,
                    label=f"M* ({analysis.stationary_point_kind})",
                )
            ax_contour.set_title(f"Траектория поиска {goal_label} на линии уровня")
            ax_contour.set_xlabel("x1")
            ax_contour.set_ylabel("x2")
            ax_contour.set_xlim(x_min, x_max)
            ax_contour.set_ylim(y_min, y_max)
            ax_contour.legend()

            values = [record.value for record in result.records if math.isfinite(record.value)]
            if not values:
                values = [result.optimum_value]
            if math.isfinite(values[0]):
                ax_convergence.plot(range(len(values)), values, "b-o", linewidth=2, markersize=4)
            else:
                ax_convergence.text(0.5, 0.5, "Нет конечных значений для графика сходимости", ha="center", va="center", transform=ax_convergence.transAxes)
            ax_convergence.set_title(f"Сходимость при поиске {goal_label}")
            ax_convergence.set_xlabel("k")
            ax_convergence.set_ylabel("F(x)")
            ax_convergence.grid(True, alpha=0.3)

            self.canvas.draw()

    def _build_plot_window(
        self,
        *,
        result: OptimizationResult,
        analysis: ExtremumAnalysis,
        fallback_points: tuple[Point2D, ...],
    ) -> tuple[float, float, float, float]:
        """Выбирает безопасное окно для графика без разлёта в inf/nan."""
        if analysis.stationary_points:
            center_x, center_y = analysis.stationary_points[0]
            span = 6.0
        else:
            finite_points = [point for point in fallback_points if math.isfinite(point[0]) and math.isfinite(point[1])]
            if finite_points:
                center_x = sum(point[0] for point in finite_points) / len(finite_points)
                center_y = sum(point[1] for point in finite_points) / len(finite_points)
                span = max(
                    max((abs(point[0] - center_x) for point in finite_points), default=1.0),
                    max((abs(point[1] - center_y) for point in finite_points), default=1.0),
                    3.0,
                ) * 2.0
            else:
                center_x, center_y = result.start_point
                span = 6.0

        span = min(max(span, 3.0), 20.0)
        half_span = span / 2.0
        return (
            center_x - half_span,
            center_x + half_span,
            center_y - half_span,
            center_y + half_span,
        )

    def _clear_plot(self) -> None:
        self.plot_context_label.hide()
        self.plot_state_label.setText("График появится после расчёта.")
        self.plot_state_label.show()
        clear_plot_canvas(self.canvas, message="График появится после запуска расчёта")

    def _selected_method(self) -> str | None:
        for method, button in self.method_buttons.items():
            if button.isChecked():
                return method
        return None

    def _set_results_empty_state(self, is_empty: bool) -> None:
        stack = self.results_empty_stack
        if stack is None:
            raise RuntimeError("EmptyStateStack не инициализирован")
        stack.set_empty(is_empty)

    def _set_busy(self, busy: bool) -> None:
        self._controls_panel.setEnabled(not busy)
        self.run_button.setDisabled(busy)
        self.run_button.setText("Считаю..." if busy else "Рассчитать")

    @staticmethod
    def _format_stop_reason(stop_reason: str) -> str:
        labels = {
            "timeout": "таймаут",
            "gradient_norm_reached": "достигнута требуемая точность",
            "no_improving_step": "не найден улучшающий шаг",
            "max_iterations_reached": "достигнут лимит итераций",
        }
        return labels.get(stop_reason, "неизвестная причина")

    def _selected_goal(self) -> str:
        for goal, button in self.goal_buttons.items():
            if button.isChecked():
                return goal
        return "max"

    def _selected_method_iterations(self) -> str:
        method = self._selected_method()
        if method is None:
            return METHOD_DEFAULTS["gradient"]["iters"]
        return METHOD_DEFAULTS[method]["iters"]

def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = GradientMethodsWindow()
    window.show()
    app.exec()
