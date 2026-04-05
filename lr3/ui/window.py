"""GUI для ЛР3: градиентные методы в стиле ЛР1/ЛР2."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
from optim_core.ui import (
    ControlsPanel,
    DarkQtThemeTokens,
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
    create_parameter_grid,
    create_primary_action_button,
    create_results_workspace,
    create_scroll_container,
    create_standard_group,
    dark_plot_context,
    set_table_data_layout,
    set_table_empty_layout,
)
from PySide6.QtCore import Qt  # type: ignore[import-not-found]
from PySide6.QtWidgets import (  # type: ignore[import-not-found]
    QApplication,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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
from lr3.domain.expression import compile_objective
from lr3.domain.models import OptimizationResult

APP_TITLE = "ЛР3 — Градиентные методы"


@dataclass(frozen=True)
class RunPayload:
    """Результат одного запуска для UI-потока."""

    expression: str
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
                background: #131824;
                border: 1px solid #3f4a62;
                border-radius: 8px;
                padding: 7px 10px;
                color: #f5f7fb;
                selection-background-color: #2379ff;
                min-height: 24px;
            }
            QLineEdit:focus { border: 1px solid #2f8fff; }
            QLabel[role="parameter-label"] {
                color: #dce6f5;
                font-size: 15px;
                font-weight: 600;
            }
            QLabel#SectionCaption {
                color: #9aa5bb;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            QLabel[role="formula-preview"] {
                background: #101827;
                border: 1px solid #304665;
                border-radius: 12px;
                padding: 10px 14px;
                color: #ecf3ff;
                font-size: 18px;
                font-weight: 700;
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
            QLabel#SectionHint {
                color: #a8b1c3;
                font-size: 12px;
            }
            QWidget#SummaryEmptyCard {
                background: #181b24;
                border: 1px solid #31384a;
                border-radius: 14px;
            }
            QTableWidget {
                background: #12161d;
                border: 1px solid #464b59;
                border-radius: 8px;
                color: #f5f7fb;
                gridline-color: #2d3241;
                selection-background-color: #2a6df4;
                alternate-background-color: #151b25;
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
                font-size: 14px;
            }
            QHeaderView::section {
                background: #222938;
                color: #dbe2ee;
                border: 0;
                border-right: 1px solid #33415b;
                border-bottom: 1px solid #33415b;
                padding: 6px 8px;
                font-size: 12px;
                font-weight: 700;
            }
            QTableCornerButton::section {
                background: #222938;
                border: 0;
                border-right: 1px solid #33415b;
                border-bottom: 1px solid #33415b;
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

        objective_group, objective_layout = create_standard_group("Целевая функция")
        expression_caption = QLabel("F(x1, x2)")
        expression_caption.setObjectName("SectionCaption")
        objective_layout.addWidget(expression_caption)

        self.expression_input = QLineEdit()
        self.expression_input.setPlaceholderText("Например: -(x1-1)^2 - 2*(x2+3)^2")
        self.expression_input.textChanged.connect(self._update_formula_preview)
        objective_layout.addWidget(self.expression_input)

        self.formula_preview = QLabel()
        self.formula_preview.setProperty("role", "formula-preview")
        self.formula_preview.setTextFormat(Qt.RichText)
        self.formula_preview.setWordWrap(True)
        self.formula_preview.setAlignment(Qt.AlignCenter)
        self.formula_preview.setMinimumHeight(74)
        objective_layout.addWidget(self.formula_preview)

        method_group, method_layout = create_standard_group("Метод")
        method_caption = QLabel("Выбор метода")
        method_caption.setObjectName("SectionCaption")
        method_layout.addWidget(method_caption)

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
        method_layout.addWidget(method_row)

        params_group = QGroupBox("Параметры")
        params_layout = create_parameter_grid(params_group)

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

        add_parameter_row(params_layout, row=0, label="Стартовая точка", control=start_row)
        add_parameter_row(params_layout, row=1, label="Точность ε", control=self.epsilon_input)
        add_parameter_row(params_layout, row=2, label="Лимит итераций", control=self.max_iterations_input)
        add_parameter_row(params_layout, row=3, label="Начальный шаг", control=self.initial_step_input)
        add_parameter_row(params_layout, row=4, label="Таймаут (сек)", control=self.timeout_input)
        add_parameter_row(params_layout, row=5, label="Мин. шаг", control=self.min_step_input)
        add_parameter_row(params_layout, row=6, label="Шаг градиента", control=self.gradient_step_input)
        add_parameter_row(
            params_layout,
            row=7,
            label="Лимит расширений шага",
            control=self.max_step_expansions_input,
        )

        self.run_button = create_primary_action_button(text="Рассчитать", on_click=self._run_clicked)

        layout.addWidget(objective_group)
        layout.addWidget(method_group)
        layout.addWidget(params_group)
        layout.addWidget(self.run_button)
        layout.addStretch(1)
        return panel

    def _build_results_panel(self) -> QWidget:
        workspace = create_results_workspace(
            results_title="Таблицы",
            plot_title="Графики",
            with_tables_empty_state=True,
            tables_empty_title="Пока нет результатов",
            tables_empty_description=(
                "Слева выбери метод, формулу и параметры запуска.\n"
                "После расчёта здесь появятся итог и таблица итераций."
            ),
            tables_empty_hint="Нажми «Рассчитать», чтобы получить результат.",
        )

        self.results_empty_stack = workspace.tables_empty_stack
        if self.results_empty_stack is None:
            raise RuntimeError("Ожидался EmptyStateStack для вкладки таблиц")

        summary_group = QGroupBox("Итог запуска")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_table = QTableWidget(0, 8)
        self.summary_table.setHorizontalHeaderLabels(
            ["Метод", "Trace ID", "Старт", "x*", "f(x*)", "N", "Успех", "Причина"]
        )
        summary_header = MathHeaderView(Qt.Horizontal, self.summary_table)
        self.summary_table.setHorizontalHeader(summary_header)
        summary_header.set_math_labels(["Метод", "Trace ID", "Старт", "x*", "f(x*)", "N", "Успех", "Причина"])
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.summary_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.summary_table.setTextElideMode(Qt.ElideNone)
        configure_data_table(
            self.summary_table,
            min_row_height=31,
            allow_selection=False,
            allow_editing=False,
            word_wrap=False,
        )
        self._set_summary_table_empty_layout()
        summary_layout.addWidget(self.summary_table)
        workspace.tables_layout.addWidget(summary_group)

        steps_group = QGroupBox("Итерации")
        steps_layout = QVBoxLayout(steps_group)
        self.steps_table = QTableWidget(0, 6)
        self.steps_table.setHorizontalHeaderLabels(["k", "x1", "x2", "F(x)", "||grad||", "step"])
        steps_header = MathHeaderView(Qt.Horizontal, self.steps_table)
        self.steps_table.setHorizontalHeader(steps_header)
        steps_header.set_math_labels(["k", "x<sub>1</sub>", "x<sub>2</sub>", "F(x)", "||grad||", "step"])
        self.steps_table.verticalHeader().setVisible(False)
        self.steps_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.steps_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.steps_table.setTextElideMode(Qt.ElideNone)
        configure_data_table(
            self.steps_table,
            min_row_height=31,
            allow_selection=False,
            allow_editing=False,
            word_wrap=False,
        )
        self._set_steps_table_empty_layout()
        steps_layout.addWidget(self.steps_table)
        workspace.tables_layout.addWidget(steps_group)

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

    def _set_method_defaults(self, method: str, checked: bool) -> None:
        if not checked:
            return
        defaults = {
            "gradient": {
                "expr": DEFAULT_GRADIENT_EXPRESSION,
                "x1": "0",
                "x2": "0",
                "step": "0.1",
                "eps": "1e-6",
                "iters": "1000",
                "timeout": "3.0",
            },
            "conjugate": {
                "expr": DEFAULT_CONJUGATE_EXPRESSION,
                "x1": "1",
                "x2": "1",
                "step": "0.2",
                "eps": "1e-6",
                "iters": "300",
                "timeout": "3.0",
            },
        }
        config = defaults[method]
        self.expression_input.setText(config["expr"])
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

        return RunPayload(expression=expression, result=result, metrics=metrics)

    def _on_run_succeeded(self, payload: object) -> None:
        self._set_busy(False)
        if not isinstance(payload, RunPayload):
            QMessageBox.critical(self, "Ошибка расчета", "Некорректный формат ответа вычислений.")
            return

        self._last_payload = payload
        self._render_summary(payload)
        self._render_iterations(payload.result)
        self._render_plot(payload)
        self._set_results_empty_state(False)

    def _on_run_failed(self, message: str, _stack: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Ошибка расчета", message)

    def _render_summary(self, payload: RunPayload) -> None:
        result = payload.result
        rows = [
            result.method_name,
            payload.metrics.trace_id,
            f"({result.start_point[0]:.6f}; {result.start_point[1]:.6f})",
            f"({result.optimum_point[0]:.8f}; {result.optimum_point[1]:.8f})",
            f"{result.optimum_value:.8f}",
            str(result.iterations_count),
            "да" if result.success else "нет",
            result.stop_reason,
        ]

        self.summary_table.setRowCount(1)
        for column, value in enumerate(rows):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)
            self.summary_table.setItem(0, column, item)
        self._set_summary_table_data_layout()

    def _render_iterations(self, result: OptimizationResult) -> None:
        records = result.records
        self.steps_table.setRowCount(len(records))

        for row, record in enumerate(records):
            grad_norm = math.hypot(record.gradient[0], record.gradient[1])
            values = (
                str(record.k),
                f"{record.point[0]:.8f}",
                f"{record.point[1]:.8f}",
                f"{record.value:.8f}",
                f"{grad_norm:.4e}",
                f"{record.step_size:.8f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.steps_table.setItem(row, column, item)

        self._set_steps_table_data_layout()

    def _render_plot(self, payload: RunPayload) -> None:
        objective = compile_objective(payload.expression)
        result = payload.result

        self.plot_state_label.hide()
        self.plot_context_label.setText(
            f"Метод: {result.method_name} | Trace ID: {payload.metrics.trace_id} | "
            f"Итераций: {result.iterations_count} | Причина остановки: {result.stop_reason}"
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

            figure.tight_layout()
            self.canvas.draw()

    def _clear_plot(self) -> None:
        self.plot_context_label.hide()
        self.plot_state_label.setText("График появится после расчёта.")
        self.plot_state_label.show()
        clear_plot_canvas(
            self.canvas,
            message="График появится после запуска расчета",
        )

    def _selected_method(self) -> str | None:
        for method, button in self.method_buttons.items():
            if button.isChecked():
                return method
        return None

    def _set_summary_table_empty_layout(self) -> None:
        set_table_empty_layout(self.summary_table)

    def _set_summary_table_data_layout(self) -> None:
        set_table_data_layout(self.summary_table, [150, 110, 170, 170, 120, 58, 84, 180])

    def _set_steps_table_empty_layout(self) -> None:
        set_table_empty_layout(self.steps_table)

    def _set_steps_table_data_layout(self) -> None:
        set_table_data_layout(self.steps_table, [60, 120, 120, 130, 110, 110])

    def _set_results_empty_state(self, is_empty: bool) -> None:
        stack = self.results_empty_stack
        if stack is None:
            raise RuntimeError("EmptyStateStack не инициализирован")
        stack.set_empty(is_empty)

    def _set_busy(self, busy: bool) -> None:
        self._controls_panel.setEnabled(not busy)
        self.run_button.setDisabled(busy)
        self.run_button.setText("Считаю..." if busy else "Рассчитать")

    def _update_formula_preview(self, raw_expression: str) -> None:
        expression = raw_expression.strip() or "—"
        readable = expression.replace("**", "^").replace("*", "·")
        self.formula_preview.setText(f"F(x<sub>1</sub>, x<sub>2</sub>) = <code>{readable}</code>")


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = GradientMethodsWindow()
    window.show()
    app.exec()
