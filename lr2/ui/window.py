"""GUI для ЛР2: метод Розенброка с непрерывным шагом."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401  # регистрация 3d-проекции
from optim_core.parsing import parse_localized_float
from optim_core.ui import (
    BatchRunUiController,
    BatchRunUiHooks,
    ControlsPanel,
    DarkQtThemeTokens,
    DynamicSeriesInputRow,
    MathHeaderView,
    PlotCanvas,
    TaskController,
    add_parameter_row,
    build_choice_chip_styles,
    build_dark_qt_base_styles,
    build_dynamic_series_styles,
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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lr2.application.services import (
    VARIANT_PRESETS,
    build_polynomial,
    parse_epsilons,
    parse_points,
    run_batch,
    run_discrete_batch,
)
from lr2.domain.models import BatchResult, SolverResult
from lr2.domain.polynomial import evaluate_polynomial, format_polynomial

APP_TITLE = "ЛР2 — Метод Розенброка"
COEFFICIENT_MAX_DEGREE = 4
COEFFICIENT_MATRIX_SIZE = COEFFICIENT_MAX_DEGREE + 1
EPSILON_INPUT_WIDTH = 96
START_INPUT_WIDTH = 64
CONTROL_BUTTON_SIZE = 44
ROW_CONTROL_SPACING = 4
ARTIFACTS_BASE_DIR = Path("report") / "lr2_runs"
PRESET_CONFIGS = {
    "variant_f1": {
        "label": "F1",
        "tooltip": "F1 (вариант 2)",
        "formula_text": "F1(x) = 9x1^4 - 6x1^2*x2 + 10x2^2 + 4x1^2 - 12x1*x2",
        "formula_display": "F<sub>1</sub>(x) = 9x<sub>1</sub><sup>4</sup> - 6x<sub>1</sub><sup>2</sup>x<sub>2</sub>"
        " + 10x<sub>2</sub><sup>2</sup><br>+ 4x<sub>1</sub><sup>2</sup> - 12x<sub>1</sub>x<sub>2</sub>",
        "starts": "0;1",
    },
    "variant_f2": {
        "label": "F2",
        "tooltip": "F2 (вариант 2)",
        "formula_text": "F2(x) = 9x1^2 + 16x2^2 - 90x1 - 128x2",
        "formula_display": "F<sub>2</sub>(x) = 9x<sub>1</sub><sup>2</sup> + 16x<sub>2</sub><sup>2</sup>"
        " - 90x<sub>1</sub> - 128x<sub>2</sub>",
        "starts": "0;0",
    },
    "custom": {
        "label": "Пользовательская",
        "tooltip": "Пользовательская функция",
        "formula_text": "Пользовательский полином",
        "formula_display": "f(x<sub>1</sub>, x<sub>2</sub>) = &Sigma; c<sub>ij</sub> &middot; "
        "x<sub>1</sub><sup>i</sup> &middot; x<sub>2</sub><sup>j</sup>",
        "starts": "0;0",
    },
}


class RosenbrockWindow(QMainWindow):
    """Главное окно приложения ЛР2."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1530, 960)
        self._apply_styles()

        self._batch_result: BatchResult | None = None
        self._selected_run_index: int | None = None
        self._active_preset_key = "variant_f1"
        self._solver_mode = "continuous"
        self._plot_mode: str = "contour"
        self._epsilon_row: DynamicSeriesInputRow | None = None
        self._start_row: DynamicSeriesInputRow | None = None
        self._delta_step_input: QLineEdit | None = None
        self._alpha_input: QLineEdit | None = None
        self._beta_input: QLineEdit | None = None
        self._run_task = TaskController(self)
        self._run_task.succeeded.connect(self._on_run_succeeded)
        self._run_task.failed.connect(self._on_run_failed)
        self._run_flow = BatchRunUiController[BatchResult](
            BatchRunUiHooks(
                assign_report=self._assign_batch_result,
                reset_selection=self._reset_batch_selection,
                render_overview=self._render_batch_overview,
                select_first=self._select_first_run_after_apply,
                clear_details=self._clear_batch_details,
            )
        )

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

        self.preset_buttons["variant_f1"].setChecked(True)
        self._on_preset_selected("variant_f1", True)
        self.mode_buttons["continuous"].setChecked(True)
        self._on_solver_mode_selected("continuous", True)
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
            QLabel[role="hint"] {
                color: #a8b1c3;
                font-size: 12px;
            }
            QSplitter::handle {
                background: #2a3549;
                border-radius: 3px;
            }
            """
            + build_choice_chip_styles()
            + build_dynamic_series_styles(separator_role="start-separator")
        )

    def _build_controls_panel(self) -> QWidget:
        controls: ControlsPanel = create_controls_panel(min_width=500, max_width=560, spacing=12)
        panel = controls.panel
        layout = controls.layout

        source_group, source_layout = create_standard_group("Функция")

        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        preset_keys = ("variant_f1", "variant_f2", "custom")
        preset_row, preset_buttons = create_choice_chip_grid(
            group=self.preset_group,
            options=tuple((PRESET_CONFIGS[key]["label"], key) for key in preset_keys),
            columns=len(preset_keys),
            horizontal_spacing=6,
            vertical_spacing=6,
            on_clicked=self._on_preset_selected,
            tooltips={key: PRESET_CONFIGS[key]["tooltip"] for key in preset_keys},
        )
        self.preset_buttons = {key: button for key, button in zip(preset_keys, preset_buttons, strict=True)}
        variant_label = QLabel("Вариант")
        variant_label.setObjectName("SectionCaption")
        source_layout.addWidget(variant_label)
        source_layout.addWidget(preset_row)

        self.formula_preview = QLabel()
        self.formula_preview.setProperty("role", "formula-preview")
        self.formula_preview.setMinimumHeight(88)
        self.formula_preview.setWordWrap(True)
        self.formula_preview.setAlignment(Qt.AlignCenter)
        self.formula_preview.setTextFormat(Qt.RichText)
        formula_label = QLabel("Формула")
        formula_label.setObjectName("SectionCaption")
        source_layout.addWidget(formula_label)
        source_layout.addWidget(self.formula_preview)

        self.coefficients_table = QTableWidget(COEFFICIENT_MATRIX_SIZE, COEFFICIENT_MATRIX_SIZE)
        self.coefficients_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.coefficients_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.coefficients_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.coefficients_table.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.coefficients_table.horizontalHeader().setDefaultSectionSize(96)
        self.coefficients_table.horizontalHeader().setMinimumSectionSize(72)
        self.coefficients_table.horizontalHeader().setFixedHeight(34)
        self.coefficients_table.verticalHeader().setDefaultSectionSize(34)
        self.coefficients_table.verticalHeader().setMinimumSectionSize(34)
        self.coefficients_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.coefficients_table.setMinimumHeight(220)
        configure_data_table(
            self.coefficients_table,
            min_row_height=30,
            allow_selection=False,
            allow_editing=True,
            word_wrap=False,
        )
        self._reset_coefficient_table()
        coeff_label = QLabel("Коэффициенты c<sub>ij</sub>")
        coeff_label.setTextFormat(Qt.RichText)
        coeff_label.setObjectName("SectionCaption")
        source_layout.addWidget(coeff_label)
        source_layout.addWidget(self.coefficients_table)

        info_group = QGroupBox("Параметры расчёта")
        info_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        info_grid = create_parameter_grid(info_group)

        self._epsilon_row = DynamicSeriesInputRow(
            add_role="series-add",
            remove_role="series-remove",
            field_role="series-item",
            placeholders=("ε",),
            field_widths=(EPSILON_INPUT_WIDTH,),
            control_button_size=CONTROL_BUTTON_SIZE,
            row_control_spacing=ROW_CONTROL_SPACING,
        )
        self._epsilon_row.add_item("0.1")

        self._start_row = DynamicSeriesInputRow(
            add_role="series-add",
            remove_role="series-remove",
            field_role="series-item",
            placeholders=("x1", "x2"),
            field_widths=(START_INPUT_WIDTH, START_INPUT_WIDTH),
            control_button_size=CONTROL_BUTTON_SIZE,
            row_control_spacing=ROW_CONTROL_SPACING,
            separator_text="—",
            separator_role="start-separator",
        )
        self._start_row.add_item("0", "1")

        add_parameter_row(info_grid, row=0, label="Точности ε", control=self._epsilon_row.row_widget)
        add_parameter_row(info_grid, row=1, label="Стартовые точки", control=self._start_row.row_widget)

        mode_group, _mode_layout = create_standard_group("Режим")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        mode_row, mode_buttons = create_choice_chip_grid(
            group=self.mode_group,
            options=(("Непрерывный", "continuous"), ("Дискретный", "discrete")),
            columns=2,
            horizontal_spacing=6,
            vertical_spacing=6,
            on_clicked=self._on_solver_mode_selected,
        )
        self.mode_buttons = {key: button for key, button in zip(("continuous", "discrete"), mode_buttons, strict=True)}
        _mode_layout.addWidget(mode_row)

        discrete_group = QGroupBox("Параметры дискретного шага")
        self._delta_step_input = QLineEdit("0.2")
        self._alpha_input = QLineEdit("1.4")
        self._beta_input = QLineEdit("-0.2")
        for widget in (self._delta_step_input, self._alpha_input, self._beta_input):
            widget.setMinimumWidth(96)
        discrete_grid = create_parameter_grid(discrete_group, label_min_width=124)
        add_parameter_row(discrete_grid, row=0, label="Δ0", control=self._delta_step_input)
        add_parameter_row(discrete_grid, row=1, label="α", control=self._alpha_input)
        add_parameter_row(discrete_grid, row=2, label="β", control=self._beta_input)
        self._discrete_group = discrete_group

        run_button = create_primary_action_button(text="Рассчитать", on_click=self._run_clicked)

        layout.addWidget(source_group)
        layout.addWidget(info_group)
        layout.addWidget(mode_group)
        layout.addWidget(discrete_group)
        layout.addWidget(run_button)
        layout.addStretch(1)
        discrete_group.setVisible(False)
        return panel

    def _build_results_panel(self) -> QWidget:
        workspace = create_results_workspace(
            results_title="Таблицы",
            plot_title="Графики",
            with_tables_empty_state=True,
            tables_empty_title="Пока нет результатов",
            tables_empty_description=(
                "Слева выбери функцию, матрицу коэффициентов и параметры расчёта.\n"
                "После запуска здесь появятся результаты и таблица итераций."
            ),
            tables_empty_hint="Нажми «Рассчитать», чтобы получить результаты и графики.",
        )
        panel = workspace.panel
        table_content_layout = workspace.tables_layout
        self.results_tab_stack = workspace.tables_empty_stack
        if self.results_tab_stack is None:
            raise RuntimeError("Ожидался EmptyStateStack для вкладки таблиц")

        summary_group = QGroupBox("Результаты расчёта")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_table = QTableWidget(0, 7)
        self.summary_table.setHorizontalHeaderLabels(
            ["№", "ε", "Старт", "x*", "f(x*)", "Итераций", "Причина завершения"]
        )
        summary_header = MathHeaderView(Qt.Horizontal, self.summary_table)
        self.summary_table.setHorizontalHeader(summary_header)
        summary_header.set_math_labels(["№", "&epsilon;", "Старт", "x*", "f(x*)", "Итераций", "Причина"])
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.summary_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.summary_table.setTextElideMode(Qt.ElideNone)
        self._set_summary_table_empty_layout()
        self.summary_table.setMinimumHeight(140)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.itemSelectionChanged.connect(self._on_summary_selection_changed)
        configure_data_table(
            self.summary_table,
            min_row_height=31,
            allow_selection=True,
            allow_editing=False,
            word_wrap=False,
        )
        summary_layout.addWidget(self.summary_table)
        table_content_layout.addWidget(summary_group)

        steps_group = QGroupBox("Итерации")
        steps_layout = QVBoxLayout(steps_group)
        self.steps_table = QTableWidget(0, 10)
        self.steps_header = MathHeaderView(Qt.Horizontal, self.steps_table)
        self.steps_table.setHorizontalHeader(self.steps_header)
        self.steps_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.steps_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.steps_table.setTextElideMode(Qt.ElideNone)
        self._set_steps_table_empty_layout()
        self.steps_table.setMinimumHeight(190)
        self.steps_table.verticalHeader().setVisible(False)
        configure_data_table(
            self.steps_table,
            min_row_height=31,
            allow_selection=False,
            allow_editing=False,
            word_wrap=False,
        )
        steps_layout.addWidget(self.steps_table)
        table_content_layout.addWidget(steps_group)
        self._set_steps_table_headers()
        self._set_results_tab_empty_state(True)

        plot_group = QGroupBox("Графики")
        plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout = QVBoxLayout(plot_group)

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(QLabel("Режим"))
        self.plot_mode_group = QButtonGroup(self)
        self.plot_mode_group.setExclusive(True)
        mode_keys = ("contour", "surface")
        mode_row_widget, mode_buttons = create_choice_chip_grid(
            group=self.plot_mode_group,
            options=(("Контуры 2D", "contour"), ("Поверхность 3D", "surface")),
            columns=len(mode_keys),
            horizontal_spacing=8,
            vertical_spacing=8,
            on_clicked=self._on_plot_mode_selected,
        )
        self.plot_mode_buttons = {key: button for key, button in zip(mode_keys, mode_buttons, strict=True)}
        mode_layout.addWidget(mode_row_widget)
        mode_layout.addStretch(1)

        self.canvas = PlotCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(620)
        plot_layout.addWidget(mode_row)
        plot_layout.addWidget(self.canvas)
        workspace.plots_layout.addWidget(plot_group)
        self.plot_mode_buttons["contour"].setChecked(True)
        self._sync_plot_mode_button_styles()
        return panel

    def _on_preset_selected(self, preset_key: str, checked: bool = True) -> None:
        if not checked:
            return
        self._apply_preset(preset_key)
        for key, button in self.preset_buttons.items():
            button.setChecked(key == preset_key)

    def _on_solver_mode_selected(self, mode_key: str, checked: bool = True) -> None:
        if not checked:
            return
        self._solver_mode = mode_key
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode_key)
        self._sync_solver_mode_ui()

    def _sync_solver_mode_ui(self) -> None:
        self._discrete_group.setVisible(self._solver_mode == "discrete")
        self._set_steps_table_headers()

    def _set_formula_preview(self, formula_text: str) -> None:
        self.formula_preview.setText(formula_text)

    def _apply_preset(self, preset_key: str) -> None:
        self._active_preset_key = preset_key
        self._set_formula_preview(PRESET_CONFIGS[preset_key]["formula_display"])
        self._set_start_points_raw(PRESET_CONFIGS[preset_key]["starts"])

        if preset_key == "custom":
            return

        matrix = VARIANT_PRESETS[preset_key]
        self._set_coefficient_matrix(matrix)

    def _reset_coefficient_table(self) -> None:
        self.coefficients_table.setHorizontalHeaderLabels(
            [f"x2^{degree}" for degree in range(COEFFICIENT_MATRIX_SIZE)]
        )
        horizontal_header = MathHeaderView(Qt.Horizontal, self.coefficients_table)
        self.coefficients_table.setHorizontalHeader(horizontal_header)
        horizontal_header.set_math_labels(
            [f"x<sub>2</sub><sup>{degree}</sup>" for degree in range(COEFFICIENT_MATRIX_SIZE)]
        )
        self.coefficients_table.setVerticalHeaderLabels(
            [f"x1^{degree}" for degree in range(COEFFICIENT_MATRIX_SIZE)]
        )
        vertical_header = MathHeaderView(Qt.Vertical, self.coefficients_table)
        self.coefficients_table.setVerticalHeader(vertical_header)
        vertical_header.set_math_labels(
            [f"x<sub>1</sub><sup>{degree}</sup>" for degree in range(COEFFICIENT_MATRIX_SIZE)]
        )
        self._set_coefficient_matrix(tuple())

    def _set_coefficient_matrix(self, matrix: tuple[tuple[float, ...], ...]) -> None:
        for i in range(COEFFICIENT_MATRIX_SIZE):
            for j in range(COEFFICIENT_MATRIX_SIZE):
                value = 0.0
                if i < len(matrix) and j < len(matrix[i]):
                    value = matrix[i][j]
                item = QTableWidgetItem(f"{value:g}")
                item.setTextAlignment(Qt.AlignCenter)
                self.coefficients_table.setItem(i, j, item)
        self.coefficients_table.resizeColumnsToContents()
        for j in range(self.coefficients_table.columnCount()):
            width = max(self.coefficients_table.columnWidth(j), 90)
            self.coefficients_table.setColumnWidth(j, width)
        self.coefficients_table.horizontalScrollBar().setValue(0)

    def _read_coefficient_matrix(self) -> tuple[tuple[float, ...], ...]:
        matrix: list[tuple[float, ...]] = []
        for i in range(self.coefficients_table.rowCount()):
            row: list[float] = []
            for j in range(self.coefficients_table.columnCount()):
                item = self.coefficients_table.item(i, j)
                raw = item.text().strip() if item and item.text().strip() else "0"
                row.append(parse_localized_float(raw, f"c[{i}][{j}]"))
            matrix.append(tuple(row))
        return tuple(matrix)

    def _run_clicked(self) -> None:
        if self._run_task.is_running():
            return
        try:
            matrix = self._read_coefficient_matrix()
            polynomial = build_polynomial(PRESET_CONFIGS[self._active_preset_key]["formula_text"], matrix)
            epsilons = parse_epsilons(self._collect_epsilons_raw())
            starts = parse_points(self._collect_start_points_raw())
            discrete_params = self._collect_discrete_parameters() if self._solver_mode == "discrete" else None
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка ввода", str(exc))
            return

        self._set_busy(True)
        if discrete_params is None:
            self._run_task.start(
                "Выполняется расчёт...",
                lambda: run_batch(polynomial, epsilons, starts),
            )
        else:
            self._run_task.start(
                "Выполняется расчёт...",
                lambda: run_discrete_batch(polynomial, epsilons, starts, **discrete_params),
            )

    def _on_run_succeeded(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 2:
            self._set_busy(False)
            return
        batch_result, metrics = payload
        if not isinstance(batch_result, BatchResult):
            self._set_busy(False)
            return
        self._run_flow.apply(batch_result)
        if hasattr(metrics, "trace_id"):
            try:
                self._save_artifacts(batch_result, trace_id=metrics.trace_id)
            except Exception as exc:
                QMessageBox.warning(self, "Сохранение артефактов", f"Не удалось сохранить артефакты: {exc}")
        self._set_busy(False)

    def _on_run_failed(self, message: str, _stack: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Ошибка расчёта", message)

    def _assign_batch_result(self, batch_result: BatchResult) -> None:
        self._batch_result = batch_result

    def _reset_batch_selection(self) -> None:
        self._selected_run_index = None

    def _render_batch_overview(self, batch_result: BatchResult) -> None:
        self._fill_summary_table(batch_result)

    def _select_first_run_after_apply(self, batch_result: BatchResult) -> bool:
        if not batch_result.runs:
            return False
        self._select_run(0, sync_table_selection=True)
        return True

    def _clear_batch_details(self) -> None:
        self._set_results_tab_empty_state(True)
        self.steps_table.setRowCount(0)
        self._set_steps_table_empty_layout()
        self._clear_plot()

    def _select_run(self, run_index: int, sync_table_selection: bool) -> None:
        """Выбирает прогон и синхронно обновляет итерации и график."""
        if self._batch_result is None:
            return
        if run_index < 0 or run_index >= len(self._batch_result.runs):
            return

        self._selected_run_index = run_index
        run = self._batch_result.runs[run_index]
        if sync_table_selection:
            self.summary_table.selectRow(run_index)
        self._fill_steps_table(run)
        self._draw_run_plot(self._batch_result, run)

    def _fill_summary_table(self, batch_result: BatchResult) -> None:
        self.summary_table.setRowCount(len(batch_result.runs))
        for index, run in enumerate(batch_result.runs):
            row = [
                str(index + 1),
                f"{run.epsilon:.6g}",
                self._format_point(run.start_point),
                self._format_point(run.optimum_point),
                f"{run.optimum_value:.8g}",
                str(run.iterations_count),
                run.stop_reason,
            ]
            for col, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 6:
                    if run.success:
                        item.setForeground(Qt.GlobalColor.green)
                self.summary_table.setItem(index, col, item)
        if batch_result.runs:
            self._set_results_tab_empty_state(False)
            self._set_summary_table_data_layout()
        else:
            self._set_results_tab_empty_state(True)
            self._set_summary_table_empty_layout()

    def _collect_epsilons_raw(self) -> str:
        if self._epsilon_row is None:
            raise ValueError("Список epsilon пуст.")
        values = [row[0] for row in self._epsilon_row.rows() if row and row[0]]
        if not values:
            raise ValueError("Список epsilon пуст.")
        return ",".join(values)

    def _collect_start_points_raw(self) -> str:
        if self._start_row is None:
            raise ValueError("Список стартовых точек пуст.")
        points: list[str] = []
        for x1_value, x2_value in self._start_row.rows():
            if not x1_value and not x2_value:
                continue
            if not x1_value or not x2_value:
                raise ValueError("Каждая стартовая точка должна содержать и x1, и x2.")
            points.append(f"{x1_value};{x2_value}")
        if not points:
            raise ValueError("Список стартовых точек пуст.")
        return " | ".join(points)

    def _collect_discrete_parameters(self) -> dict[str, float]:
        if self._delta_step_input is None or self._alpha_input is None or self._beta_input is None:
            raise ValueError("Параметры дискретного шага не инициализированы.")
        return {
            "delta_step": parse_localized_float(self._delta_step_input.text().strip(), "Δ0"),
            "alpha": parse_localized_float(self._alpha_input.text().strip(), "α"),
            "beta": parse_localized_float(self._beta_input.text().strip(), "β"),
        }

    def _set_start_points_raw(self, raw: str) -> None:
        if self._start_row is None:
            return
        chunks = [chunk.strip() for chunk in raw.split("|") if chunk.strip()]
        if not chunks:
            self._start_row.reset((("", ""),))
            return
        values: list[tuple[str, str]] = []
        for chunk in chunks:
            parts = [part.strip() for part in chunk.split(";")]
            if len(parts) != 2:
                continue
            values.append((parts[0], parts[1]))
        self._start_row.reset(tuple(values) if values else (("", ""),))

    def _set_steps_table_headers(self) -> None:
        if not hasattr(self, "steps_table") or not hasattr(self, "steps_header"):
            return
        if self._solver_mode == "discrete":
            labels = [
                "K",
                "x_k",
                "F(x_k)",
                "j",
                "d_j",
                "y_j",
                "f(y_j)",
                "Δ_j",
                "y_j+Δ_j d_j",
                "f(y_j+Δ_j d_j)",
            ]
            math_labels = [
                "K",
                "x<sub>k</sub>",
                "F(x<sub>k</sub>)",
                "j",
                "d<sub>j</sub>",
                "y<sub>j</sub>",
                "f(y<sub>j</sub>)",
                "&Delta;<sub>j</sub>",
                "y<sub>j</sub>+&Delta;<sub>j</sub>d<sub>j</sub>",
                "f(y<sub>j</sub>+&Delta;<sub>j</sub>d<sub>j</sub>)",
            ]
        else:
            labels = ["K", "x_k", "F(x_k)", "j", "d_j", "y_j", "f(y_j)", "λ_j", "y_j+1", "f(y_j+1)"]
            math_labels = [
                "K",
                "x<sub>k</sub>",
                "F(x<sub>k</sub>)",
                "j",
                "d<sub>j</sub>",
                "y<sub>j</sub>",
                "f(y<sub>j</sub>)",
                "&lambda;<sub>j</sub>",
                "y<sub>j+1</sub>",
                "f(y<sub>j+1</sub>)",
            ]
        self.steps_table.setHorizontalHeaderLabels(labels)
        self.steps_header.set_math_labels(math_labels)

    def _on_summary_selection_changed(self) -> None:
        if not self._batch_result:
            return
        selected_indexes = self.summary_table.selectionModel().selectedRows()
        if not selected_indexes:
            return
        self._select_run(selected_indexes[0].row(), sync_table_selection=False)

    def _on_plot_mode_selected(self, mode_key: str, checked: bool = True) -> None:
        if not checked:
            return
        self._plot_mode = mode_key
        self._sync_plot_mode_button_styles()
        if not self._batch_result or self._selected_run_index is None:
            self._clear_plot()
            return
        if self._selected_run_index < 0 or self._selected_run_index >= len(self._batch_result.runs):
            self._clear_plot()
            return
        run = self._batch_result.runs[self._selected_run_index]
        self._draw_run_plot(self._batch_result, run)

    def _sync_plot_mode_button_styles(self) -> None:
        for mode_key, button in self.plot_mode_buttons.items():
            button.setChecked(mode_key == self._plot_mode)

    def _set_busy(self, busy: bool) -> None:
        self._controls_panel.setEnabled(not busy)

    def _set_results_tab_empty_state(self, is_empty: bool) -> None:
        if self.results_tab_stack is None:
            return
        self.results_tab_stack.set_empty(is_empty)

    def _set_summary_table_empty_layout(self) -> None:
        """Пустая таблица итогов должна занимать ширину равномерно."""
        set_table_empty_layout(self.summary_table)

    def _set_summary_table_data_layout(self) -> None:
        """Для данных: ширина по содержимому + горизонтальный скролл."""
        set_table_data_layout(self.summary_table, [48, 76, 140, 140, 110, 58, 220])

    def _set_steps_table_empty_layout(self) -> None:
        """Для пустого состояния убираем визуальный «обрубок» справа."""
        set_table_empty_layout(self.steps_table)

    def _set_steps_table_data_layout(self) -> None:
        """Для данных: ширина по содержимому + горизонтальный скролл."""
        set_table_data_layout(self.steps_table, [52, 112, 92, 44, 112, 112, 92, 92, 120, 110])

    def _fill_steps_table(self, run: SolverResult) -> None:
        self.steps_table.setRowCount(len(run.steps))
        for row_idx, step in enumerate(run.steps):
            cells = [
                str(step.k),
                self._format_point(step.x_k),
                f"{step.f_x_k:.8g}",
                str(step.j),
                self._format_point(step.direction),
                self._format_point(step.y_j),
                f"{step.f_y_j:.8g}",
                f"{step.lambda_j:.8g}",
                self._format_point(step.y_next),
                f"{step.f_y_next:.8g}",
            ]
            for col_idx, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.steps_table.setItem(row_idx, col_idx, item)
        self._set_steps_table_data_layout()

    def _draw_run_plot(self, batch_result: BatchResult, run: SolverResult) -> None:
        points = np.array(run.trajectory)
        if points.ndim != 2 or points.shape[1] != 2:
            self._clear_plot()
            return
        x_vals = points[:, 0]
        y_vals = points[:, 1]

        x_min = float(np.min(x_vals))
        x_max = float(np.max(x_vals))
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))

        span_x = max(x_max - x_min, 1.0)
        span_y = max(y_max - y_min, 1.0)
        margin_x = span_x * 0.6
        margin_y = span_y * 0.6

        grid_x = np.linspace(x_min - margin_x, x_max + margin_x, 120)
        grid_y = np.linspace(y_min - margin_y, y_max + margin_y, 120)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        mesh_z = self._evaluate_mesh(batch_result, mesh_x, mesh_y)

        with dark_plot_context():
            self.canvas.figure.clear()
            self.canvas.figure.patch.set_facecolor("#171b24")
            mode = self._plot_mode
            if mode == "surface":
                ax_surface = self.canvas.figure.add_subplot(1, 1, 1, projection="3d")
                z_clipped = self._build_surface_mesh(mesh_z)
                ax_surface.plot_surface(
                    mesh_x,
                    mesh_y,
                    z_clipped,
                    cmap="turbo",
                    alpha=0.9,
                    linewidth=0,
                    antialiased=True,
                )
                path_z = [evaluate_polynomial(batch_result.polynomial, point[0], point[1]) for point in run.trajectory]
                z_span = max(float(np.nanmax(z_clipped) - np.nanmin(z_clipped)), 1.0)
                z_offset = z_span * 0.02
                lifted_path_z = [value + z_offset for value in path_z]
                # Темный контур под основной линией для читаемости на любом colormap фоне.
                ax_surface.plot(
                    x_vals,
                    y_vals,
                    lifted_path_z,
                    color="#121722",
                    linewidth=5.2,
                    marker="o",
                    markersize=5.6,
                    markerfacecolor="#121722",
                    markeredgewidth=0.0,
                )
                ax_surface.plot(
                    x_vals,
                    y_vals,
                    lifted_path_z,
                    color="#ffffff",
                    linewidth=3.0,
                    marker="o",
                    markersize=4.3,
                    markerfacecolor="#ff4f87",
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.7,
                )
                ax_surface.scatter(
                    [x_vals[0]],
                    [y_vals[0]],
                    [lifted_path_z[0]],
                    color="#2da3ff",
                    edgecolors="#ffffff",
                    linewidths=1.0,
                    s=130,
                    depthshade=False,
                )
                ax_surface.scatter(
                    [x_vals[-1]],
                    [y_vals[-1]],
                    [lifted_path_z[-1]],
                    color="#57d773",
                    edgecolors="#ffffff",
                    linewidths=1.0,
                    s=130,
                    depthshade=False,
                )
                ax_surface.set_title("3D поверхность + траектория")
                ax_surface.set_xlabel("x1")
                ax_surface.set_ylabel("x2")
                ax_surface.set_zlabel("f(x1, x2)")
                for axis in (ax_surface.xaxis, ax_surface.yaxis, ax_surface.zaxis):
                    pane = getattr(axis, "pane", None)
                    if pane is not None:
                        pane.set_facecolor((0.12, 0.16, 0.23, 0.45))
                ax_surface.tick_params(colors="#c4cfdf")
                ax_surface.view_init(elev=26, azim=-56)
            else:
                ax_contour = self.canvas.figure.add_subplot(1, 1, 1)
                contour = ax_contour.contour(mesh_x, mesh_y, mesh_z, levels=24, cmap="turbo")
                ax_contour.set_facecolor("#10141f")
                ax_contour.clabel(contour, inline=True, fontsize=8, colors="#dce6f5")
                ax_contour.plot(
                    x_vals,
                    y_vals,
                    marker="o",
                    color="#ffffff",
                    linewidth=3.1,
                    markersize=4.5,
                    markerfacecolor="#ff4f87",
                    markeredgewidth=0.0,
                    zorder=3,
                )
                for idx, (x_item, y_item) in enumerate(zip(x_vals, y_vals, strict=True)):
                    if idx == 0 or idx == len(x_vals) - 1 or idx % 2 == 0:
                        ax_contour.annotate(
                            str(idx),
                            (x_item, y_item),
                            color="#f0f6ff",
                            fontsize=9,
                            textcoords="offset points",
                            xytext=(5, -8),
                        )
                ax_contour.scatter([x_vals[0]], [y_vals[0]], color="#2da3ff", s=100, label="Старт", zorder=4)
                ax_contour.scatter([x_vals[-1]], [y_vals[-1]], color="#57d773", s=100, label="Финиш", zorder=4)
                ax_contour.set_title("Линии уровня + траектория")
                ax_contour.set_xlabel("x1")
                ax_contour.set_ylabel("x2")
                ax_contour.set_aspect("equal", adjustable="box")
                ax_contour.grid(True)
                legend = ax_contour.legend(loc="upper right", framealpha=0.92)
                for text in legend.get_texts():
                    text.set_color("#e8f0ff")

        self.canvas.draw_idle()

    def _save_artifacts(self, batch_result: BatchResult, trace_id: str) -> Path:
        run_dir = self._create_artifacts_dir(trace_id)
        formula_text = format_polynomial(batch_result.polynomial)
        self._write_summary_csv(run_dir / "summary.csv", batch_result)
        (run_dir / "formula.txt").write_text(f"Формула: {formula_text}\n", encoding="utf-8")

        for run_index, run in enumerate(batch_result.runs, start=1):
            single_dir = run_dir / f"run_{run_index:03d}"
            single_dir.mkdir(parents=True, exist_ok=True)
            self._write_iterations_csv(single_dir / "iterations.csv", run)
            self._save_run_plot_png(batch_result, run, mode="contour", output_path=single_dir / "contour.png")
            self._save_run_plot_png(batch_result, run, mode="surface", output_path=single_dir / "surface.png")

        return run_dir

    def _create_artifacts_dir(self, trace_id: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{trace_id}"
        ARTIFACTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
        candidate = ARTIFACTS_BASE_DIR / base_name
        suffix = 1
        while candidate.exists():
            candidate = ARTIFACTS_BASE_DIR / f"{base_name}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _write_summary_csv(self, path: Path, batch_result: BatchResult) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["#", "epsilon", "start", "x*", "f(x*)", "N", "status"])
            for index, run in enumerate(batch_result.runs, start=1):
                writer.writerow(
                    [
                        index,
                        f"{run.epsilon:.6g}",
                        self._format_point(run.start_point),
                        self._format_point(run.optimum_point),
                        f"{run.optimum_value:.8g}",
                        run.iterations_count,
                        "OK" if run.success else "MAX_ITER",
                    ]
                )

    def _write_iterations_csv(self, path: Path, run: SolverResult) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["K", "x_k", "F(x_k)", "j", "d_j", "y_j", "f(y_j)", "λ_j", "y_{j+1}", "f(y_{j+1})"])
            for step in run.steps:
                writer.writerow(
                    [
                        step.k,
                        self._format_point(step.x_k),
                        f"{step.f_x_k:.8g}",
                        step.j,
                        self._format_point(step.direction),
                        self._format_point(step.y_j),
                        f"{step.f_y_j:.8g}",
                        f"{step.lambda_j:.8g}",
                        self._format_point(step.y_next),
                        f"{step.f_y_next:.8g}",
                    ]
                )

    def _save_run_plot_png(self, batch_result: BatchResult, run: SolverResult, mode: str, output_path: Path) -> None:
        points = np.array(run.trajectory)
        if points.ndim != 2 or points.shape[1] != 2:
            return

        x_vals = points[:, 0]
        y_vals = points[:, 1]
        x_min = float(np.min(x_vals))
        x_max = float(np.max(x_vals))
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        span_x = max(x_max - x_min, 1.0)
        span_y = max(y_max - y_min, 1.0)
        margin_x = span_x * 0.6
        margin_y = span_y * 0.6

        grid_x = np.linspace(x_min - margin_x, x_max + margin_x, 120)
        grid_y = np.linspace(y_min - margin_y, y_max + margin_y, 120)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        mesh_z = self._evaluate_mesh(batch_result, mesh_x, mesh_y)

        with dark_plot_context():
            figure = Figure(figsize=(9.0, 6.0), dpi=120)
            figure.patch.set_facecolor("#171b24")
            if mode == "surface":
                ax_surface = figure.add_subplot(1, 1, 1, projection="3d")
                z_clipped = self._build_surface_mesh(mesh_z)
                ax_surface.plot_surface(
                    mesh_x,
                    mesh_y,
                    z_clipped,
                    cmap="turbo",
                    alpha=0.72,
                    linewidth=0,
                    antialiased=True,
                )
                path_z = [evaluate_polynomial(batch_result.polynomial, point[0], point[1]) for point in run.trajectory]
                z_span = max(float(np.nanmax(z_clipped) - np.nanmin(z_clipped)), 1.0)
                z_offset = z_span * 0.08
                lifted_path_z = [value + z_offset for value in path_z]
                ax_surface.plot(
                    x_vals,
                    y_vals,
                    lifted_path_z,
                    color="#121722",
                    linewidth=7.0,
                    marker="o",
                    markersize=6.6,
                    markerfacecolor="#121722",
                    markeredgewidth=0.0,
                )
                ax_surface.plot(
                    x_vals,
                    y_vals,
                    lifted_path_z,
                    color="#ff2d95",
                    linewidth=4.2,
                    marker="o",
                    markersize=5.2,
                    markerfacecolor="#ff4f87",
                    markeredgecolor="#ffffff",
                    markeredgewidth=1.0,
                )
                ax_surface.scatter(
                    [x_vals[0]],
                    [y_vals[0]],
                    [lifted_path_z[0]],
                    color="#2da3ff",
                    edgecolors="#ffffff",
                    linewidths=1.0,
                    s=130,
                    depthshade=False,
                )
                ax_surface.scatter(
                    [x_vals[-1]],
                    [y_vals[-1]],
                    [lifted_path_z[-1]],
                    color="#57d773",
                    edgecolors="#ffffff",
                    linewidths=1.0,
                    s=130,
                    depthshade=False,
                )
                ax_surface.set_title("3D поверхность + траектория")
                ax_surface.set_xlabel("x1")
                ax_surface.set_ylabel("x2")
                ax_surface.set_zlabel("f(x1, x2)")
                for axis in (ax_surface.xaxis, ax_surface.yaxis, ax_surface.zaxis):
                    pane = getattr(axis, "pane", None)
                    if pane is not None:
                        pane.set_facecolor((0.12, 0.16, 0.23, 0.45))
                ax_surface.tick_params(colors="#c4cfdf")
                ax_surface.set_zlim(float(np.nanmin(z_clipped)), float(np.nanmax(z_clipped) + z_offset * 1.5))
                ax_surface.view_init(elev=34, azim=-44)
            else:
                ax_contour = figure.add_subplot(1, 1, 1)
                contour = ax_contour.contour(mesh_x, mesh_y, mesh_z, levels=24, cmap="turbo")
                ax_contour.set_facecolor("#10141f")
                ax_contour.clabel(contour, inline=True, fontsize=8, colors="#dce6f5")
                ax_contour.plot(
                    x_vals,
                    y_vals,
                    marker="o",
                    color="#ffffff",
                    linewidth=3.1,
                    markersize=4.5,
                    markerfacecolor="#ff4f87",
                    markeredgewidth=0.0,
                    zorder=3,
                )
                ax_contour.scatter([x_vals[0]], [y_vals[0]], color="#2da3ff", s=100, label="Старт", zorder=4)
                ax_contour.scatter([x_vals[-1]], [y_vals[-1]], color="#57d773", s=100, label="Финиш", zorder=4)
                ax_contour.set_title("Линии уровня + траектория")
                ax_contour.set_xlabel("x1")
                ax_contour.set_ylabel("x2")
                ax_contour.set_aspect("equal", adjustable="box")
                ax_contour.grid(True)
                legend = ax_contour.legend(loc="upper right", framealpha=0.92)
                for text in legend.get_texts():
                    text.set_color("#e8f0ff")

            figure.savefig(output_path, dpi=120, bbox_inches="tight")

    def _evaluate_mesh(self, batch_result: BatchResult, mesh_x: np.ndarray, mesh_y: np.ndarray) -> np.ndarray:
        polynomial = batch_result.polynomial
        result = np.zeros_like(mesh_x, dtype=float)
        for i, row in enumerate(polynomial.coefficients):
            x_part = np.power(mesh_x, i)
            for j, coefficient in enumerate(row):
                if coefficient == 0.0:
                    continue
                result += coefficient * x_part * np.power(mesh_y, j)
        return result

    @staticmethod
    def _build_surface_mesh(mesh_z: np.ndarray) -> np.ndarray:
        """Готовит численно устойчивую сетку для 3D-поверхности."""
        finite_values = mesh_z[np.isfinite(mesh_z)]
        if finite_values.size == 0:
            return np.zeros_like(mesh_z, dtype=float)
        low = float(np.percentile(finite_values, 5))
        high = float(np.percentile(finite_values, 95))
        if high < low:
            low, high = high, low
        clipped = np.clip(np.nan_to_num(mesh_z, nan=0.0, posinf=high, neginf=low), low, high)
        return clipped

    def _clear_plot(self) -> None:
        clear_plot_canvas(
            self.canvas,
            message="Выберите запуск из таблицы, чтобы увидеть графики",
        )

    @staticmethod
    def _format_point(point: tuple[float, ...]) -> str:
        return "(" + ", ".join(f"{value:.6g}" for value in point) + ")"


def main() -> None:
    """Запускает Qt-приложение."""
    app = QApplication(sys.argv)
    window = RosenbrockWindow()
    window.show()
    sys.exit(app.exec())
