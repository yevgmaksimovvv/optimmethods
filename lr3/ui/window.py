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
    create_parameter_grid,
    create_primary_action_button,
    create_results_workspace,
    create_scroll_container,
    create_standard_group,
    dark_plot_context,
)
from PySide6.QtCore import Qt  # type: ignore[import-not-found]
from PySide6.QtWidgets import (  # type: ignore[import-not-found]
    QApplication,
    QButtonGroup,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from lr3.application.services import (
    DEFAULT_CONJUGATE_EXPRESSION,
    DEFAULT_GRADIENT_EXPRESSION,
    ServiceMetrics,
    build_config,
    build_start_point,
    run_conjugate,
    run_gradient,
)
from lr3.domain.expression import analyze_local_extremum, compile_objective
from lr3.domain.models import ExtremumAnalysis, IterationRecord, MethodConfig, OptimizationResult, Point2D

APP_TITLE = "ЛР3 — Градиентные методы"

FUNCTION_PRESET_CONFIGS = {
    "gradient": {
        "label": "Градиентная",
        "tooltip": "Функция из задания для метода первого порядка.",
        "expression": DEFAULT_GRADIENT_EXPRESSION,
        "method": "gradient",
    },
    "conjugate": {
        "label": "Сопряжённых градиентов",
        "tooltip": "Функция из задания для метода сопряжённых градиентов.",
        "expression": DEFAULT_CONJUGATE_EXPRESSION,
        "method": "conjugate",
    },
    "custom": {
        "label": "Пользовательская",
        "tooltip": "Своя функция. Выражение можно редактировать вручную.",
        "expression": None,
        "method": None,
    },
}

METHOD_DEFAULTS = {
    "gradient": {
        "x1": "0",
        "x2": "0",
        "step": "0.1",
        "eps": "1e-6",
        "iters": "1000",
        "timeout": "3.0",
    },
    "conjugate": {
        "x1": "1",
        "x2": "1",
        "step": "0.2",
        "eps": "1e-6",
        "iters": "300",
        "timeout": "3.0",
    },
}


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
        self._custom_expression_cache = DEFAULT_GRADIENT_EXPRESSION

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
            + """
            QLineEdit {
                min-height: 24px;
            }
            QSplitter::handle {
                background: #2a3549;
                border-radius: 3px;
            }
            """
            + build_choice_chip_styles()
        )

    def _build_controls_panel(self) -> QWidget:
        controls: ControlsPanel = create_controls_panel(min_width=500, max_width=560, spacing=12)
        panel = controls.panel
        layout = controls.layout

        basic_group, basic_layout = create_standard_group("Основные")
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
            horizontal_spacing=6,
            vertical_spacing=6,
            on_clicked=self._on_preset_selected,
            tooltips={key: FUNCTION_PRESET_CONFIGS[key]["tooltip"] for key in preset_keys},
        )
        self.preset_buttons = {key: button for key, button in zip(preset_keys, preset_buttons, strict=True)}
        basic_layout.addWidget(preset_row)

        expression_caption = QLabel("Формула")
        expression_caption.setObjectName("SectionCaption")
        basic_layout.addWidget(expression_caption)

        self.expression_input = QLineEdit()
        self.expression_input.setPlaceholderText("Например: -(x1-1)^2 - 2*(x2+3)^2")
        self.expression_input.textChanged.connect(self._update_formula_preview)
        basic_layout.addWidget(self.expression_input)

        self.formula_preview = QLabel()
        self.formula_preview.setProperty("role", "formula-preview")
        self.formula_preview.setTextFormat(Qt.RichText)
        self.formula_preview.setWordWrap(True)
        self.formula_preview.setAlignment(Qt.AlignCenter)
        self.formula_preview.setMinimumHeight(74)
        basic_layout.addWidget(self.formula_preview)

        method_caption = QLabel("Метод")
        method_caption.setObjectName("SectionCaption")
        basic_layout.addWidget(method_caption)

        self.method_group = QButtonGroup(self)
        self.method_group.setExclusive(True)
        method_keys = ("gradient", "conjugate")
        method_row, method_buttons = create_choice_chip_grid(
            group=self.method_group,
            options=(("Градиентный", "gradient"), ("Сопряжённые градиенты", "conjugate")),
            columns=2,
            on_clicked=self._set_method_defaults,
        )
        self.method_buttons = {key: button for key, button in zip(method_keys, method_buttons, strict=True)}
        basic_layout.addWidget(method_row)

        self.start_x1_input = QLineEdit()
        self.start_x2_input = QLineEdit()
        self.epsilon_input = QLineEdit()
        self.max_iterations_input = QLineEdit()
        self.initial_step_input = QLineEdit()
        self.timeout_input = QLineEdit()
        self.min_step_input = QLineEdit()
        self.gradient_step_input = QLineEdit()
        self.max_step_expansions_input = QLineEdit()

        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(10)
        start_x1_caption = QLabel("x1")
        start_x1_caption.setObjectName("SectionCaption")
        start_x2_caption = QLabel("x2")
        start_x2_caption.setObjectName("SectionCaption")
        start_layout.addWidget(start_x1_caption)
        start_layout.addWidget(self.start_x1_input, 1)
        start_layout.addWidget(start_x2_caption)
        start_layout.addWidget(self.start_x2_input, 1)

        basic_layout.addWidget(start_row)

        advanced_group = QGroupBox("Дополнительные")
        advanced_layout = create_parameter_grid(advanced_group)

        add_parameter_row(advanced_layout, row=0, label="Точность ε", control=self.epsilon_input)
        add_parameter_row(advanced_layout, row=1, label="Лимит итераций", control=self.max_iterations_input)
        add_parameter_row(advanced_layout, row=2, label="Начальный шаг", control=self.initial_step_input)
        add_parameter_row(advanced_layout, row=3, label="Таймаут (сек)", control=self.timeout_input)
        add_parameter_row(advanced_layout, row=4, label="Мин. шаг", control=self.min_step_input)
        add_parameter_row(advanced_layout, row=5, label="Шаг градиента", control=self.gradient_step_input)
        add_parameter_row(
            advanced_layout,
            row=6,
            label="Лимит расширений шага",
            control=self.max_step_expansions_input,
        )

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
        self._active_preset_key = preset_key
        for key, button in self.preset_buttons.items():
            button.setChecked(key == preset_key)
        self._apply_preset(preset_key)

    def _apply_preset(self, preset_key: str) -> None:
        preset = FUNCTION_PRESET_CONFIGS[preset_key]
        expression = preset["expression"]
        is_custom = expression is None
        self.expression_input.setReadOnly(not is_custom)
        if expression is not None:
            self.expression_input.setText(expression)
        else:
            self.expression_input.setText(self._custom_expression_cache)
        if preset["method"] is not None:
            self.method_buttons[preset["method"]].setChecked(True)
            self._set_method_defaults(preset["method"], True)

    def _set_method_defaults(self, method: str, checked: bool = True) -> None:
        if not checked:
            return
        config = METHOD_DEFAULTS[method]
        self.start_x1_input.setText(config["x1"])
        self.start_x2_input.setText(config["x2"])
        self.initial_step_input.setText(config["step"])
        self.epsilon_input.setText(config["eps"])
        self.max_iterations_input.setText(config["iters"])
        self.timeout_input.setText(config["timeout"])
        self.min_step_input.setText("1e-8")
        self.gradient_step_input.setText("1e-6")
        self.max_step_expansions_input.setText("16")
        self._update_formula_preview(self.expression_input.text())

    def _run_clicked(self) -> None:
        if self._run_task.is_running():
            return
        method = self._selected_method()
        if method is None:
            QMessageBox.critical(self, "Ошибка ввода", "Выбери метод оптимизации.")
            return

        self._set_busy(True)
        self._run_task.start(
            "lr3-run",
            lambda: self._run_method(
                method=method,
                expression=self.expression_input.text().strip(),
                x1_raw=self.start_x1_input.text(),
                x2_raw=self.start_x2_input.text(),
                epsilon_raw=self.epsilon_input.text(),
                max_iterations_raw=self.max_iterations_input.text(),
                initial_step_raw=self.initial_step_input.text(),
                timeout_raw=self.timeout_input.text(),
                min_step_raw=self.min_step_input.text(),
                gradient_step_raw=self.gradient_step_input.text(),
                max_step_expansions_raw=self.max_step_expansions_input.text(),
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
        analysis = analyze_local_extremum(payload.expression, result.start_point, goal="max")

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
        self._add_report_row(form, "Цель", self._math_text("Поиск максимума"))
        self._add_report_row(form, "Функция", self._math_text(f"F(x1, x2) = {self._format_formula(payload.expression)}"))
        self._add_report_row(form, "M<sub>0</sub>", self._math_text(self._format_point(result.start_point)))
        return card

    def _build_analysis_card(
        self,
        *,
        analysis: ExtremumAnalysis,
    ) -> QWidget:
        card, form = self._create_report_card("Аналитическая подготовка")
        self._add_report_row(form, "Исходная функция", self._math_text(f"F(x1, x2) = {self._format_formula(analysis.expression)}"))
        self._add_report_row(form, "Начальная точка", self._math_text(self._format_point(analysis.start_point)))
        self._add_report_row(
            form,
            "Общий градиент",
            self._math_text(
                f"∇f(x<sub>1</sub>, x<sub>2</sub>) = ({self._format_formula(analysis.gradient_formula[0])}; {self._format_formula(analysis.gradient_formula[1])})"
            ),
        )
        self._add_report_row(
            form,
            "∇f(M<sub>0</sub>)",
            self._math_text(f"({self._format_scalar(analysis.gradient_at_start[0])}; {self._format_scalar(analysis.gradient_at_start[1])})"),
        )
        if analysis.stationary_points:
            self._add_report_row(
                form,
                "Решение ∇f(x) = 0",
                self._math_text(self._format_stationary_points(analysis.stationary_points)),
            )
            if analysis.stationary_gradient is not None:
                self._add_report_row(
                    form,
                    "∇f(M<sub>*</sub>)",
                    self._math_text(
                        f"({self._format_scalar(analysis.stationary_gradient[0])}; {self._format_scalar(analysis.stationary_gradient[1])})"
                    ),
                )
        else:
            self._add_report_row(form, "Решение ∇f(x) = 0", self._math_text("строгий аналитический поиск стационарной точки не выполнен"))
        self._add_report_row(form, "Матрица Гессе", self._math_text(self._format_matrix(analysis.hessian_formula)))
        if analysis.hessian_at_stationary_point is not None:
            self._add_report_row(
                form,
                "H(M<sub>*</sub>)",
                self._math_text(self._format_matrix(analysis.hessian_at_stationary_point, scalar_formatter=self._format_scalar)),
            )
        else:
            self._add_report_row(form, "H(M<sub>*</sub>)", self._math_text("не вычислена"))
        self._add_report_row(form, "Вывод", self._math_text(analysis.theory_conclusion))
        self._add_report_row(form, "Согласование", self._math_text(analysis.goal_alignment))
        self._add_report_row(form, "Ограничение", self._math_text(analysis.strictness_note))
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
        for record in result.records:
            has_step = record.step_size > 0.0
            new_point = self._translate_point(record.point, record.gradient, record.step_size) if has_step else None
            new_value = objective(new_point) if new_point is not None else None
            note = self._gradient_step_note(record.step_size, payload.config.initial_step, result)
            card, form = self._create_report_card(f"Итерация {record.k + 1}", object_name="IterationCard")
            self._add_report_row(form, "Текущая точка", self._math_text(self._format_point(record.point)))
            self._add_report_row(form, "F(M<sub>k</sub>)", self._math_text(self._format_scalar(record.value)))
            self._add_report_row(
                form,
                "∇f(M<sub>k</sub>)",
                self._math_text(f"({self._format_scalar(record.gradient[0])}; {self._format_scalar(record.gradient[1])})"),
            )
            self._add_report_row(form, "h<sub>k</sub>", self._math_text(self._format_step(record.step_size)))
            if new_point is not None and new_value is not None:
                self._add_report_row(
                    form,
                    f"M<sub>{record.k + 1}</sub>",
                    self._math_text(
                        f"M<sub>{record.k + 1}</sub> = M<sub>{record.k}</sub> + h<sub>{record.k}</sub>·∇f(M<sub>{record.k}</sub>) = {self._format_point(new_point)}"
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
                "∇f(M<sub>k</sub>)",
                self._math_text(f"({self._format_scalar(record.gradient[0])}; {self._format_scalar(record.gradient[1])})"),
            )
            if record.direction is not None:
                self._add_report_row(
                    form,
                    "s<sub>k</sub>",
                    self._math_text(
                        f"({self._format_scalar(record.direction[0])}; {self._format_scalar(record.direction[1])})"
                    ),
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
        card, form = self._create_report_card("Итоговый вывод")
        self._add_report_row(form, "Требовалось найти", self._math_text("поиск максимума"))
        self._add_report_row(form, "Теория", self._math_text(analysis.stationary_point_kind))
        self._add_report_row(form, "Согласование", self._math_text(analysis.goal_alignment))
        self._add_report_row(form, "Численный статус", self._math_text(self._format_success_text(result.success)))
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
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card_layout = QFormLayout(card)
        card_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        card_layout.setFormAlignment(Qt.AlignTop)
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
    def _format_matrix(
        matrix: tuple[tuple[str, str], tuple[str, str]] | tuple[tuple[float, float], tuple[float, float]],
        *,
        scalar_formatter: Callable[[float], str] | None = None,
    ) -> str:
        formatter = scalar_formatter or (lambda value: value if isinstance(value, str) else str(value))
        first_row = " ; ".join(formatter(value) for value in matrix[0])
        second_row = " ; ".join(formatter(value) for value in matrix[1])
        return f"[[{first_row}]; [{second_row}]]"

    @staticmethod
    def _math_text(text: str) -> str:
        return f"<span style=\"font-family: 'Consolas', 'SF Mono', monospace;\">{text}</span>"

    def _render_plot(self, payload: RunPayload) -> None:
        objective = compile_objective(payload.expression)
        result = payload.result

        self.plot_state_label.hide()
        self.plot_context_label.setText(
            f"Метод: {self._format_method_title(result.method_name)} | Итераций: {result.iterations_count} | "
            f"Причина завершения: {self._format_stop_reason(result.stop_reason)}"
        )
        self.plot_context_label.show()

        with dark_plot_context():
            figure = self.canvas.figure
            figure.clear()
            figure.patch.set_facecolor("#171b24")

            ax_contour = figure.add_subplot(121)
            ax_convergence = figure.add_subplot(122)

            xs = [item.point[0] for item in result.records]
            ys = [item.point[1] for item in result.records]
            if not xs:
                xs = [result.start_point[0], result.optimum_point[0]]
                ys = [result.start_point[1], result.optimum_point[1]]

            margin = 1.0
            x_min = min(xs) - margin
            x_max = max(xs) + margin
            y_min = min(ys) - margin
            y_max = max(ys) + margin

            grid_x = np.linspace(x_min, x_max, 80)
            grid_y = np.linspace(y_min, y_max, 80)
            x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)
            z_mesh = np.zeros_like(x_mesh)

            for i in range(x_mesh.shape[0]):
                for j in range(x_mesh.shape[1]):
                    z_mesh[i, j] = objective((float(x_mesh[i, j]), float(y_mesh[i, j])))

            contour = ax_contour.contourf(x_mesh, y_mesh, z_mesh, levels=25, cmap="viridis")
            figure.colorbar(contour, ax=ax_contour)
            ax_contour.plot(xs, ys, "r.-", linewidth=2, markersize=6)
            ax_contour.scatter([xs[0]], [ys[0]], c="white", edgecolors="black", label="start")
            ax_contour.scatter([result.optimum_point[0]], [result.optimum_point[1]], c="red", label="optimum")
            ax_contour.set_title("Траектория на линии уровня")
            ax_contour.set_xlabel("x1")
            ax_contour.set_ylabel("x2")
            ax_contour.legend()

            values = [record.value for record in result.records]
            if not values:
                values = [result.optimum_value]
            ax_convergence.plot(range(len(values)), values, "b-o", linewidth=2, markersize=4)
            ax_convergence.set_title("Сходимость")
            ax_convergence.set_xlabel("k")
            ax_convergence.set_ylabel("F(x)")
            ax_convergence.grid(True, alpha=0.3)

            self.canvas.draw()

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

    def _update_formula_preview(self, raw_expression: str) -> None:
        expression = raw_expression.strip() or "—"
        if self._active_preset_key == "custom" and raw_expression.strip():
            self._custom_expression_cache = raw_expression.strip()
        readable = expression.replace("**", "^").replace("*", "·")
        self.formula_preview.setText(f"F(x<sub>1</sub>, x<sub>2</sub>) = <code>{readable}</code>")


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = GradientMethodsWindow()
    window.show()
    app.exec()
