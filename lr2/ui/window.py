"""GUI для ЛР2: метод Розенброка с непрерывным шагом."""

from __future__ import annotations

import math
import sys

import numpy as np
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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lr2.application.artifacts import RosenbrockArtifactsStore
from lr2.application.services import (
    VARIANT_PRESETS,
    build_polynomial,
    parse_epsilons,
    parse_points,
    run_batch,
    run_discrete_batch,
)
from lr2.domain.models import (
    BATCH_STATUS_DOMAIN_REFUSAL,
    BATCH_STATUS_SUCCESS,
    BATCH_STATUS_UNEXPECTED_ERROR,
    BatchItemResult,
    BatchResult,
    SolverResult,
)
from lr2.domain.polynomial import evaluate_polynomial

APP_TITLE = "ЛР2 — Метод Розенброка"
COEFFICIENT_MAX_DEGREE = 4
COEFFICIENT_MATRIX_SIZE = COEFFICIENT_MAX_DEGREE + 1
EPSILON_INPUT_WIDTH = 96
START_INPUT_WIDTH = 64
CONTROL_BUTTON_SIZE = 36
ROW_CONTROL_SPACING = 2
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


def _surface_view_angles(points: np.ndarray) -> tuple[float, float]:
    """Подбирает ракурс 3D-графика по геометрии траектории."""
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return 30.0, -55.0

    xy = np.asarray(points[:, :2], dtype=float)
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    if not np.isfinite(centered).all():
        return 30.0, -55.0

    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return 30.0, -55.0

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    azim = float(np.degrees(np.arctan2(principal[1], principal[0]))) + 90.0
    if azim > 180.0:
        azim -= 360.0
    if azim <= -180.0:
        azim += 360.0

    spans = np.ptp(xy, axis=0)
    major = float(max(spans.max(), 1.0))
    minor = float(max(spans.min(), 1e-6))
    aspect = major / minor
    elev = 30.0 + min(10.0, max(0.0, math.log1p(max(aspect - 1.0, 0.0)) * 4.0))
    return elev, azim


def _draw_surface_trajectory(
    ax_surface: _Axes3D,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    lifted_path_z: list[float],
) -> None:
    """Рисует траекторию поверх поверхности с контрастной обводкой."""
    ax_surface.plot(
        x_vals,
        y_vals,
        lifted_path_z,
        color="#0f131a",
        linewidth=12.0,
        marker="o",
        markersize=6.6,
        markerfacecolor="#0f131a",
        markeredgewidth=0.0,
        alpha=1.0,
    )
    ax_surface.plot(
        x_vals,
        y_vals,
        lifted_path_z,
        color="#ff2d95",
        linewidth=6.8,
        marker="o",
        markersize=5.4,
        markerfacecolor="#ff4f87",
        markeredgecolor="#ffffff",
        markeredgewidth=1.0,
        alpha=1.0,
    )
    ax_surface.plot(
        x_vals,
        y_vals,
        lifted_path_z,
        color="#ffffff",
        linewidth=3.4,
        marker="o",
        markersize=4.6,
        markerfacecolor="#ff4f87",
        markeredgecolor="#ffffff",
        markeredgewidth=0.8,
        alpha=1.0,
    )
    ax_surface.scatter(
        [x_vals[0]],
        [y_vals[0]],
        [lifted_path_z[0]],
        color="#2da3ff",
        edgecolors="#ffffff",
        linewidths=1.0,
        s=160,
        depthshade=False,
        alpha=1.0,
    )
    ax_surface.scatter(
        [x_vals[-1]],
        [y_vals[-1]],
        [lifted_path_z[-1]],
        color="#57d773",
        edgecolors="#ffffff",
        linewidths=1.0,
        s=160,
        depthshade=False,
        alpha=1.0,
    )


def _report_surface_aspect(x_span: float, y_span: float, z_span: float) -> tuple[float, float, float]:
    """Подбирает масштаб осей для отчетного 3D-графика."""
    xy_span = max(x_span, y_span, 1.0)
    z_display_span = min(max(z_span * 0.35, xy_span * 0.35), xy_span * 1.25)
    return xy_span, xy_span, z_display_span


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
        self._artifacts_store = RosenbrockArtifactsStore()
        self.results_tabs: QTabWidget | None = None
        self.results_tab_indexes = None
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
                min-height: 22px;
                padding: 6px 10px;
                font-size: 14px;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#SectionCaption {
                font-size: 11px;
            }
            QLabel[role="parameter-label"] {
                font-size: 14px;
                font-weight: 600;
            }
            QLabel[role="formula-preview"] {
                font-size: 16px;
                padding: 8px 12px;
                min-height: 82px;
            }
            QPushButton[variant="primary"] {
                min-height: 38px;
                padding: 7px 14px;
                font-size: 14px;
            }
            QPushButton[role="series-add"] {
                font-size: 18px;
                padding-bottom: 1px;
            }
            QPushButton[role="series-remove"] {
                font-size: 15px;
                padding-bottom: 1px;
            }
            QLineEdit[role="series-item"] {
                padding-top: 0px;
                padding-bottom: 0px;
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
        info_grid = create_parameter_grid(
            info_group,
            label_min_width=136,
            horizontal_spacing=8,
            vertical_spacing=7,
            margins=(16, 16, 16, 12),
        )

        self._epsilon_row = DynamicSeriesInputRow(
            add_role="series-add",
            remove_role="series-remove",
            field_role="series-item",
            placeholders=("ε",),
            field_widths=(EPSILON_INPUT_WIDTH,),
            control_button_size=CONTROL_BUTTON_SIZE,
            row_control_spacing=ROW_CONTROL_SPACING,
            scroll_min_height=56,
            scroll_max_height=60,
            container_min_height=42,
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
            scroll_min_height=56,
            scroll_max_height=60,
            container_min_height=42,
        )
        self._start_row.add_item("0", "1")

        epsilon_label = add_parameter_row(info_grid, row=0, label="Точности ε", control=self._epsilon_row.row_widget)
        start_label = add_parameter_row(info_grid, row=1, label="Стартовые точки", control=self._start_row.row_widget)
        epsilon_label.setMinimumWidth(132)
        start_label.setMinimumWidth(132)

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
        mode_hint = QLabel("Параметры дискретного шага доступны только в дискретном режиме.")
        mode_hint.setObjectName("SectionHint")
        mode_hint.setWordWrap(True)
        _mode_layout.addWidget(mode_hint)

        discrete_group = QGroupBox("Параметры дискретного шага")
        self._delta_step_input = QLineEdit("0.2")
        self._alpha_input = QLineEdit("1.4")
        self._beta_input = QLineEdit("-0.2")
        for widget in (self._delta_step_input, self._alpha_input, self._beta_input):
            widget.setMinimumWidth(96)
        discrete_grid = create_parameter_grid(
            discrete_group,
            label_min_width=112,
            horizontal_spacing=8,
            vertical_spacing=7,
            margins=(16, 16, 16, 12),
        )
        delta_label = add_parameter_row(discrete_grid, row=0, label="Δ0", control=self._delta_step_input)
        alpha_label = add_parameter_row(discrete_grid, row=1, label="α", control=self._alpha_input)
        beta_label = add_parameter_row(discrete_grid, row=2, label="β", control=self._beta_input)
        for label in (delta_label, alpha_label, beta_label):
            label.setMinimumWidth(112)
        self._discrete_group = discrete_group

        run_button = create_primary_action_button(text="Рассчитать", on_click=self._run_clicked, min_height=36)

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
        self.results_tabs = workspace.tabs
        self.results_tab_indexes = workspace.tab_indexes
        table_content_layout = workspace.tables_layout
        self.results_tab_stack = workspace.tables_empty_stack
        if self.results_tab_stack is None:
            raise RuntimeError("Ожидался EmptyStateStack для вкладки таблиц")
        self.results_tabs.currentChanged.connect(self._on_results_tab_changed)

        summary_group = QGroupBox("Результаты расчёта")
        summary_layout = QVBoxLayout(summary_group)
        self.batch_summary_label = QLabel("Пока нет результатов.")
        self.batch_summary_label.setObjectName("SectionHint")
        self.batch_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.batch_summary_label)
        self.summary_table = QTableWidget(0, 8)
        self.summary_table.setHorizontalHeaderLabels(
            ["№", "ε", "Старт", "x*", "f(x*)", "Итераций", "Статус", "Сообщение"]
        )
        summary_header = MathHeaderView(Qt.Horizontal, self.summary_table)
        self.summary_table.setHorizontalHeader(summary_header)
        summary_header.set_math_labels(
            ["№", "&epsilon;", "Старт", "x*", "f(x*)", "Итераций", "Статус", "Сообщение"]
        )
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.summary_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.summary_table.setTextElideMode(Qt.ElideNone)
        set_table_empty_layout(self.summary_table)
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
        self.steps_state_label = QLabel("Выберите запуск, чтобы увидеть итерации.")
        self.steps_state_label.setObjectName("SectionHint")
        self.steps_state_label.setWordWrap(True)
        steps_layout.addWidget(self.steps_state_label)
        self.steps_table = QTableWidget(0, 10)
        self.steps_header = MathHeaderView(Qt.Horizontal, self.steps_table)
        self.steps_table.setHorizontalHeader(self.steps_header)
        self.steps_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.steps_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.steps_table.setTextElideMode(Qt.ElideNone)
        set_table_empty_layout(self.steps_table)
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
        for button in self.plot_mode_buttons.values():
            button.setMinimumWidth(168)
        mode_row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        is_discrete = self._solver_mode == "discrete"
        self._discrete_group.setVisible(is_discrete)
        self._discrete_group.setEnabled(is_discrete)
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
                self._artifacts_store.save_batch_result(batch_result, trace_id=metrics.trace_id)
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
        summary = batch_result.summary
        self.batch_summary_label.setText(
            "Комбинаций: "
            f"{summary.total_count}, "
            f"успехов: {summary.success_count}, "
            f"доменный отказ: {summary.domain_refusal_count}, "
            f"неожиданных ошибок: {summary.unexpected_error_count}"
        )
        self._fill_summary_table(batch_result)

    def _select_first_run_after_apply(self, batch_result: BatchResult) -> bool:
        batch_items = self._batch_items(batch_result)
        if not batch_items:
            return False
        self._select_run(0, sync_table_selection=True)
        return True

    def _clear_batch_details(self) -> None:
        self._set_results_tab_empty_state(True)
        self.steps_table.setRowCount(0)
        set_table_empty_layout(self.steps_table)
        self._clear_plot()

    def _select_run(self, run_index: int, sync_table_selection: bool) -> None:
        """Выбирает прогон и синхронно обновляет итерации и график."""
        if self._batch_result is None:
            return
        batch_items = self._batch_items(self._batch_result)
        if run_index < 0 or run_index >= len(batch_items):
            return

        self._selected_run_index = run_index
        item = batch_items[run_index]
        if sync_table_selection:
            self.summary_table.blockSignals(True)
            try:
                self.summary_table.selectRow(run_index)
            finally:
                self.summary_table.blockSignals(False)
        self._show_run_details(item)

    def _show_run_details(self, item: BatchItemResult) -> None:
        if item.run is None:
            self.steps_table.setRowCount(0)
            set_table_empty_layout(self.steps_table)
            self.steps_state_label.setText(
                item.message or "У выбранной комбинации нет результата расчёта."
            )
            self.steps_state_label.show()
            self._clear_plot(message=item.message or "У выбранной комбинации нет результата расчёта.")
            return

        run = item.run
        if run.steps:
            self.steps_state_label.hide()
            self._fill_steps_table(run)
        else:
            self.steps_table.setRowCount(0)
            set_table_empty_layout(self.steps_table)
            self.steps_state_label.setText("У выбранного запуска нет сохранённой истории итераций.")
            self.steps_state_label.show()
        self._draw_run_plot(self._batch_result, run)

    def _fill_summary_table(self, batch_result: BatchResult) -> None:
        batch_items = self._batch_items(batch_result)
        self.summary_table.setRowCount(len(batch_items))
        for index, item in enumerate(batch_items):
            run = item.run
            if run is None:
                row = [
                    str(index + 1),
                    f"{item.epsilon:.6g}",
                    self._format_point(item.start_point),
                    "",
                    "",
                    "",
                    self._format_batch_status(item),
                    item.message or "",
                ]
            else:
                row = [
                    str(index + 1),
                    f"{run.epsilon:.6g}",
                    self._format_point(run.start_point),
                    self._format_point(run.optimum_point),
                    f"{run.optimum_value:.8g}",
                    str(run.iterations_count),
                    self._format_batch_status(item),
                    run.stop_reason,
                ]
            for col, value in enumerate(row):
                table_item = QTableWidgetItem(value)
                table_item.setTextAlignment(Qt.AlignCenter)
                if col == 6:
                    self._style_status_item(table_item, item)
                self.summary_table.setItem(index, col, table_item)
        if batch_items:
            self._set_results_tab_empty_state(False)
            set_table_data_layout(self.summary_table, [48, 76, 140, 140, 110, 58, 148, 320])
        else:
            self._set_results_tab_empty_state(True)
            set_table_empty_layout(self.summary_table)

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

    def _on_results_tab_changed(self, index: int) -> None:
        if self._batch_result is None or self._selected_run_index is None or self.results_tab_indexes is None:
            return
        batch_items = self._batch_items(self._batch_result)
        if self._selected_run_index < 0 or self._selected_run_index >= len(batch_items):
            return
        if index == self.results_tab_indexes.plot:
            item = batch_items[self._selected_run_index]
            if item.run is None:
                self._clear_plot(message=item.message or "У выбранной комбинации нет результата расчёта.")
            else:
                self._draw_run_plot(self._batch_result, item.run)
        elif index == self.results_tab_indexes.results:
            self._select_run(self._selected_run_index, sync_table_selection=True)

    def _on_plot_mode_selected(self, mode_key: str, checked: bool = True) -> None:
        if not checked:
            return
        self._plot_mode = mode_key
        self._sync_plot_mode_button_styles()
        if not self._batch_result or self._selected_run_index is None:
            self._clear_plot()
            return
        batch_items = self._batch_items(self._batch_result)
        if self._selected_run_index < 0 or self._selected_run_index >= len(batch_items):
            self._clear_plot()
            return
        item = batch_items[self._selected_run_index]
        if item.run is None:
            self._clear_plot(message=item.message or "У выбранной комбинации нет результата расчёта.")
            return
        self._draw_run_plot(self._batch_result, item.run)

    def _sync_plot_mode_button_styles(self) -> None:
        for mode_key, button in self.plot_mode_buttons.items():
            button.setChecked(mode_key == self._plot_mode)

    def _set_busy(self, busy: bool) -> None:
        self._controls_panel.setEnabled(not busy)

    def _set_results_tab_empty_state(self, is_empty: bool) -> None:
        if self.results_tab_stack is None:
            return
        self.results_tab_stack.set_empty(is_empty)

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
        set_table_data_layout(self.steps_table, [52, 112, 92, 44, 112, 112, 92, 92, 120, 110])
        self.steps_state_label.hide()

    def _batch_items(self, batch_result: BatchResult) -> tuple[BatchItemResult, ...]:
        if batch_result.items:
            return batch_result.items
        return tuple(
            BatchItemResult(
                epsilon=run.epsilon,
                start_point=run.start_point,
                status=BATCH_STATUS_SUCCESS,
                run=run,
            )
            for run in batch_result.runs
        )

    @staticmethod
    def _format_batch_status(item: BatchItemResult) -> str:
        if item.status == BATCH_STATUS_SUCCESS:
            return "OK"
        if item.status == BATCH_STATUS_DOMAIN_REFUSAL:
            return "Доменный отказ"
        if item.status == BATCH_STATUS_UNEXPECTED_ERROR:
            return "Ошибка"
        return item.status

    @staticmethod
    def _style_status_item(table_item: QTableWidgetItem, batch_item: BatchItemResult) -> None:
        if batch_item.status == BATCH_STATUS_SUCCESS:
            color = Qt.GlobalColor.green if batch_item.run is None or batch_item.run.success else Qt.GlobalColor.yellow
            table_item.setForeground(color)
        elif batch_item.status == BATCH_STATUS_DOMAIN_REFUSAL:
            table_item.setForeground(Qt.GlobalColor.darkYellow)
        elif batch_item.status == BATCH_STATUS_UNEXPECTED_ERROR:
            table_item.setForeground(Qt.GlobalColor.red)

    def _draw_run_plot(self, batch_result: BatchResult, run: SolverResult) -> None:
        points = np.array(run.trajectory)
        if points.ndim != 2 or points.shape[1] != 2 or points.size == 0:
            self._clear_plot(message="У выбранного запуска нет данных для графика.")
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
        mesh_z = self._artifacts_store.build_mesh(batch_result, mesh_x, mesh_y)

        with dark_plot_context():
            self.canvas.figure.clear()
            self.canvas.figure.patch.set_facecolor("#171b24")
            mode = self._plot_mode
            if mode == "surface":
                ax_surface = self.canvas.figure.add_subplot(1, 1, 1, projection="3d")
                if hasattr(ax_surface, "set_proj_type"):
                    ax_surface.set_proj_type("ortho")
                z_clipped = self._build_surface_mesh(mesh_z)
                ax_surface.plot_surface(
                    mesh_x,
                    mesh_y,
                    z_clipped,
                    cmap="turbo",
                    alpha=0.86,
                    linewidth=0,
                    antialiased=True,
                )
                path_z = [evaluate_polynomial(batch_result.polynomial, point[0], point[1]) for point in run.trajectory]
                z_span = max(float(np.nanmax(z_clipped) - np.nanmin(z_clipped)), 1.0)
                lifted_path_z = [value + z_span * 0.02 for value in path_z]
                _draw_surface_trajectory(ax_surface, x_vals, y_vals, lifted_path_z)
                ax_surface.set_title("Поверхность и траектория")
                ax_surface.set_xlabel("x1")
                ax_surface.set_ylabel("x2")
                ax_surface.set_zlabel("f(x1, x2)")
                ax_surface.grid(False)
                spans = _report_surface_aspect(x_max - x_min, y_max - y_min, z_span)
                if hasattr(ax_surface, "set_box_aspect"):
                    ax_surface.set_box_aspect(spans)
                for axis in (ax_surface.xaxis, ax_surface.yaxis, ax_surface.zaxis):
                    pane = getattr(axis, "pane", None)
                    if pane is not None:
                        pane.set_facecolor((0.12, 0.16, 0.23, 0.45))
                ax_surface.tick_params(colors="#c4cfdf")
                elev, azim = _surface_view_angles(points)
                ax_surface.view_init(elev=elev, azim=azim)
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

        self.canvas.draw()

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

    def _clear_plot(self, message: str = "Выберите запуск из таблицы, чтобы увидеть графики") -> None:
        clear_plot_canvas(
            self.canvas,
            message=message,
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
