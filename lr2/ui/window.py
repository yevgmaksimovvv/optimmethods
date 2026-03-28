"""GUI для ЛР2: метод Розенброка с непрерывным шагом."""

from __future__ import annotations

import sys

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QPainter, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyleOptionHeader,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lr2.application.services import (
    VARIANT_PRESETS,
    build_polynomial,
    parse_epsilons,
    parse_float,
    parse_points,
    run_batch,
)
from lr2.domain.models import BatchResult, SolverResult
from lr2.domain.polynomial import evaluate_polynomial, format_polynomial

APP_TITLE = "ЛР2 — Метод Розенброка (непрерывный шаг)"
COEFFICIENT_MAX_DEGREE = 4
COEFFICIENT_MATRIX_SIZE = COEFFICIENT_MAX_DEGREE + 1
EPSILON_INPUT_WIDTH = 96
START_INPUT_WIDTH = 64
CONTROL_BUTTON_SIZE = 44
ROW_CONTROL_SPACING = 4
START_SEPARATOR_WIDTH = 12
MATPLOTLIB_DARK_RC = {
    "figure.facecolor": "#171b24",
    "axes.facecolor": "#10141f",
    "axes.edgecolor": "#4a5974",
    "axes.labelcolor": "#dce6f5",
    "axes.titlecolor": "#e8f0ff",
    "xtick.color": "#c4cfdf",
    "ytick.color": "#c4cfdf",
    "grid.color": "#2c3b55",
    "grid.alpha": 0.35,
    "text.color": "#dce6f5",
    "legend.facecolor": "#161d2b",
    "legend.edgecolor": "#4a5974",
}
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


class MplCanvas(FigureCanvasQTAgg):
    """Встраиваемый matplotlib canvas."""

    def __init__(self, parent: QWidget | None = None):
        self.figure = Figure(figsize=(13.0, 8.0), dpi=100)
        self.figure.set_constrained_layout(True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.patch.set_facecolor("#171b24")


class MathHeaderView(QHeaderView):
    """Header, который рендерит подписи через HTML для математики."""

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None = None):
        super().__init__(orientation, parent)
        self._labels: list[str] = []

    def set_math_labels(self, labels: list[str]) -> None:
        self._labels = labels
        self.viewport().update()

    def paintSection(self, painter: QPainter, rect: QRect, logical_index: int) -> None:
        if logical_index < 0:
            return

        option = QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.section = logical_index
        option.text = ""
        option.position = QStyleOptionHeader.SectionPosition.Middle
        option.textAlignment = Qt.AlignCenter
        self.style().drawControl(QStyle.ControlElement.CE_Header, option, painter, self)

        if logical_index >= len(self._labels):
            return
        text = self._labels[logical_index]
        if not text:
            return

        text_rect = rect.adjusted(6, 2, -6, -2)
        doc = QTextDocument(self)
        doc.setHtml(
            "<div style='text-align:center; color:#dde8fa; font-weight:700; font-size:12px;'>"
            f"{text}</div>"
        )
        doc.setTextWidth(text_rect.width())
        content_height = doc.size().height()
        top_shift = max((text_rect.height() - content_height) / 2.0, 0.0)
        painter.save()
        painter.translate(text_rect.left(), text_rect.top() + top_shift)
        doc.drawContents(painter, QRectF(0, 0, text_rect.width(), text_rect.height()))
        painter.restore()


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

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        controls = self._build_controls_panel()
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        controls_scroll.setWidget(controls)

        results = self._build_results_panel()

        splitter.addWidget(controls_scroll)
        splitter.addWidget(results)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([510, 1020])
        splitter.setHandleWidth(8)

        self.preset_buttons["variant_f1"].setChecked(True)
        self._apply_preset("variant_f1")
        self._clear_plot()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #1b1f2a;
                color: #f0f2f5;
                font-family: "Segoe UI", "Helvetica Neue", "Arial", sans-serif;
                font-size: 15px;
            }
            QGroupBox {
                border: 1px solid #3f4a62;
                border-radius: 12px;
                margin-top: 12px;
                padding: 13px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #dde5f3;
            }
            QLabel { background: transparent; }
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
            QPushButton {
                background: #2b3447;
                border: 1px solid #4b5873;
                border-radius: 8px;
                padding: 9px 14px;
                color: #f5f7fb;
                font-weight: 600;
                min-height: 22px;
            }
            QPushButton:hover { background: #36415a; }
            QPushButton:pressed { background: #242d3d; }
            QPushButton[variant="primary"] {
                background: #0f7aff;
                border-color: #3b94ff;
            }
            QPushButton[variant="primary"]:hover { background: #2588ff; }
            QPushButton[variant="primary"]:pressed { background: #0d66d8; }
            QPushButton[role="preset"] {
                min-height: 32px;
                min-width: 0px;
                padding: 8px 10px;
                border-radius: 9px;
                background: #253149;
                border: 1px solid #46567a;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton[role="preset"][checked="true"] {
                background: #0f7aff;
                border-color: #3b94ff;
            }
            QPushButton[role="epsilon-add"] {
                min-height: 44px;
                max-height: 44px;
                min-width: 44px;
                max-width: 44px;
                border-radius: 8px;
                font-size: 22px;
                font-weight: 700;
                padding: 0 0 2px 0;
                background: #0f7aff;
                border-color: #3b94ff;
            }
            QPushButton[role="epsilon-add"]:hover { background: #2588ff; }
            QPushButton[role="epsilon-add"]:pressed { background: #0d66d8; }
            QPushButton[role="start-add"] {
                min-height: 44px;
                max-height: 44px;
                min-width: 44px;
                max-width: 44px;
                border-radius: 8px;
                font-size: 22px;
                font-weight: 700;
                padding: 0 0 2px 0;
                background: #0f7aff;
                border-color: #3b94ff;
            }
            QPushButton[role="start-add"]:hover { background: #2588ff; }
            QPushButton[role="start-add"]:pressed { background: #0d66d8; }
            QPushButton[role="epsilon-remove"] {
                min-height: 44px;
                max-height: 44px;
                min-width: 44px;
                max-width: 44px;
                border-radius: 7px;
                font-size: 18px;
                font-weight: 700;
                padding: 0 0 2px 0;
                background: #2b3447;
                border-color: #4b5873;
            }
            QPushButton[role="epsilon-remove"]:hover { background: #36415a; }
            QPushButton[role="epsilon-remove"]:pressed { background: #242d3d; }
            QPushButton[role="epsilon-remove"]:disabled {
                background: #1f2533;
                color: #66738d;
                border-color: #3c465f;
            }
            QPushButton[role="start-remove"] {
                min-height: 44px;
                max-height: 44px;
                min-width: 44px;
                max-width: 44px;
                border-radius: 7px;
                font-size: 18px;
                font-weight: 700;
                padding: 0 0 2px 0;
                background: #2b3447;
                border-color: #4b5873;
            }
            QPushButton[role="start-remove"]:hover { background: #36415a; }
            QPushButton[role="start-remove"]:pressed { background: #242d3d; }
            QPushButton[role="start-remove"]:disabled {
                background: #1f2533;
                color: #66738d;
                border-color: #3c465f;
            }
            QLineEdit[role="epsilon-item"] {
                font-weight: 700;
                text-align: center;
                min-height: 44px;
                padding-top: 0px;
                padding-bottom: 0px;
            }
            QLineEdit[role="start-item"] {
                font-weight: 700;
                text-align: center;
                min-height: 44px;
                padding-top: 0px;
                padding-bottom: 0px;
            }
            QLabel[role="start-separator"] {
                color: #a9b8d4;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel[role="hint"] {
                color: #a8b1c3;
                font-size: 12px;
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
            QTableWidget {
                background: #111723;
                border: 1px solid #3f4a62;
                border-radius: 8px;
                color: #f5f7fb;
                gridline-color: #273247;
                selection-background-color: #1a5fcc;
                alternate-background-color: #151d2a;
                font-size: 14px;
            }
            QHeaderView::section {
                background: #1d2a3f;
                color: #dde8fa;
                border: 0;
                border-right: 1px solid #304665;
                border-bottom: 1px solid #304665;
                padding: 7px 8px;
                font-weight: 700;
                font-size: 12px;
            }
            QTableCornerButton::section {
                background: #1d2a3f;
                border: 0;
                border-right: 1px solid #304665;
                border-bottom: 1px solid #304665;
            }
            QSplitter::handle {
                background: #2a3549;
                border-radius: 3px;
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
            """
        )

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(500)
        panel.setMaximumWidth(560)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        source_group = QGroupBox("Функция")
        source_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        source_layout = QVBoxLayout(source_group)
        source_layout.setContentsMargins(16, 16, 16, 14)
        source_layout.setSpacing(8)

        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        self.preset_buttons: dict[str, QPushButton] = {}
        for preset_key in ("variant_f1", "variant_f2", "custom"):
            button = QPushButton(PRESET_CONFIGS[preset_key]["label"])
            button.setCheckable(True)
            button.setProperty("role", "preset")
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setToolTip(PRESET_CONFIGS[preset_key]["tooltip"])
            button.clicked.connect(lambda checked, key=preset_key: self._on_preset_selected(key, checked))
            self.preset_group.addButton(button)
            self.preset_buttons[preset_key] = button
            preset_layout.addWidget(button)
        source_layout.addWidget(QLabel("Вариант"))
        source_layout.addWidget(preset_row)

        self.formula_preview = QLabel()
        self.formula_preview.setProperty("role", "formula-preview")
        self.formula_preview.setMinimumHeight(100)
        self.formula_preview.setWordWrap(True)
        self.formula_preview.setAlignment(Qt.AlignCenter)
        self.formula_preview.setTextFormat(Qt.RichText)
        source_layout.addWidget(QLabel("Формула"))
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
        self._configure_table(self.coefficients_table, min_row_height=30)
        self._reset_coefficient_table()
        coeff_label = QLabel("Коэффициенты c<sub>ij</sub>")
        coeff_label.setTextFormat(Qt.RichText)
        source_layout.addWidget(coeff_label)
        source_layout.addWidget(self.coefficients_table)

        info_group = QGroupBox("Параметры расчетов")
        info_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(16, 16, 16, 14)
        info_layout.setSpacing(8)

        self.epsilon_items: list[tuple[QWidget, QLineEdit, QPushButton]] = []
        epsilon_row = QWidget()
        epsilon_row.setMinimumHeight(76)
        epsilon_row_layout = QHBoxLayout(epsilon_row)
        epsilon_row_layout.setContentsMargins(0, 0, 0, 0)
        epsilon_row_layout.setSpacing(8)
        epsilon_row_layout.setAlignment(Qt.AlignVCenter)

        self.epsilon_scroll = QScrollArea()
        self.epsilon_scroll.setWidgetResizable(False)
        self.epsilon_scroll.setFrameShape(QScrollArea.NoFrame)
        self.epsilon_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.epsilon_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.epsilon_scroll.setMinimumHeight(72)
        self.epsilon_scroll.setMaximumHeight(76)
        self.epsilon_scroll.setMinimumWidth(0)
        self.epsilon_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        self.epsilon_fields_container = QWidget()
        self.epsilon_fields_layout = QHBoxLayout(self.epsilon_fields_container)
        self.epsilon_fields_layout.setContentsMargins(0, 0, 0, 0)
        self.epsilon_fields_layout.setSpacing(8)
        self.epsilon_fields_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.epsilon_fields_layout.setSizeConstraint(QHBoxLayout.SetFixedSize)
        self.epsilon_scroll.setWidget(self.epsilon_fields_container)

        self.add_epsilon_button = QPushButton("+")
        self.add_epsilon_button.setProperty("role", "epsilon-add")
        self.add_epsilon_button.setFixedSize(CONTROL_BUTTON_SIZE, CONTROL_BUTTON_SIZE)
        self.add_epsilon_button.clicked.connect(lambda _checked=False: self._add_epsilon_input())

        epsilon_row_layout.addWidget(self.epsilon_scroll, 1, Qt.AlignTop)
        epsilon_row_layout.addWidget(self.add_epsilon_button, 0, Qt.AlignTop)
        epsilon_row_layout.setStretch(0, 1)
        epsilon_row_layout.setStretch(1, 0)

        self._add_epsilon_input("0.1")

        self.start_point_items: list[tuple[QWidget, QLineEdit, QLineEdit, QPushButton]] = []
        start_row = QWidget()
        start_row.setMinimumHeight(76)
        start_row_layout = QHBoxLayout(start_row)
        start_row_layout.setContentsMargins(0, 0, 0, 0)
        start_row_layout.setSpacing(8)
        start_row_layout.setAlignment(Qt.AlignVCenter)

        self.start_points_scroll = QScrollArea()
        self.start_points_scroll.setWidgetResizable(False)
        self.start_points_scroll.setFrameShape(QScrollArea.NoFrame)
        self.start_points_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.start_points_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.start_points_scroll.setMinimumHeight(72)
        self.start_points_scroll.setMaximumHeight(76)
        self.start_points_scroll.setMinimumWidth(0)
        self.start_points_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        self.start_points_container = QWidget()
        self.start_points_layout = QHBoxLayout(self.start_points_container)
        self.start_points_layout.setContentsMargins(0, 0, 0, 0)
        self.start_points_layout.setSpacing(8)
        self.start_points_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.start_points_layout.setSizeConstraint(QHBoxLayout.SetFixedSize)
        self.start_points_scroll.setWidget(self.start_points_container)

        self.add_start_point_button = QPushButton("+")
        self.add_start_point_button.setProperty("role", "start-add")
        self.add_start_point_button.setFixedSize(CONTROL_BUTTON_SIZE, CONTROL_BUTTON_SIZE)
        self.add_start_point_button.clicked.connect(lambda _checked=False: self._add_start_point_input())

        start_row_layout.addWidget(self.start_points_scroll, 1, Qt.AlignTop)
        start_row_layout.addWidget(self.add_start_point_button, 0, Qt.AlignTop)
        start_row_layout.setStretch(0, 1)
        start_row_layout.setStretch(1, 0)

        self._add_start_point_input("0", "1")

        epsilon_label = QLabel("Точности ε")
        epsilon_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        info_layout.addWidget(epsilon_label)
        info_layout.addWidget(epsilon_row)
        starts_label = QLabel("Стартовые точки")
        starts_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        info_layout.addWidget(starts_label)
        info_layout.addWidget(start_row)

        run_button = QPushButton("Рассчитать")
        run_button.setProperty("variant", "primary")
        run_button.clicked.connect(self._run_clicked)
        run_button.setMinimumHeight(42)

        layout.addWidget(source_group)
        layout.addWidget(info_group)
        layout.addWidget(run_button)
        layout.addStretch(1)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.tabBar().setExpanding(True)
        tabs.tabBar().setUsesScrollButtons(False)
        layout.addWidget(tabs)

        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(12)

        summary_group = QGroupBox("Итоги запусков")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_table = QTableWidget(0, 7)
        self.summary_table.setHorizontalHeaderLabels(["#", "ε", "Старт", "x*", "f(x*)", "N", "Статус"])
        summary_header = MathHeaderView(Qt.Horizontal, self.summary_table)
        self.summary_table.setHorizontalHeader(summary_header)
        summary_header.set_math_labels(["#", "&epsilon;", "Старт", "x*", "f(x*)", "N", "Статус"])
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.summary_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.summary_table.setTextElideMode(Qt.ElideNone)
        self._set_summary_table_empty_layout()
        self.summary_table.setMinimumHeight(140)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.itemSelectionChanged.connect(self._on_summary_selection_changed)
        self._configure_table(self.summary_table, min_row_height=31)
        summary_layout.addWidget(self.summary_table)
        table_layout.addWidget(summary_group)

        steps_group = QGroupBox("Итерации")
        steps_layout = QVBoxLayout(steps_group)
        self.steps_table = QTableWidget(0, 10)
        self.steps_table.setHorizontalHeaderLabels(
            ["K", "x_k", "F(x_k)", "j", "d_j", "y_j", "f(y_j)", "λ_j", "y_j+1", "f(y_j+1)"]
        )
        steps_header = MathHeaderView(Qt.Horizontal, self.steps_table)
        self.steps_table.setHorizontalHeader(steps_header)
        steps_header.set_math_labels(
            [
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
        )
        self.steps_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.steps_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.steps_table.setTextElideMode(Qt.ElideNone)
        self._set_steps_table_empty_layout()
        self.steps_table.setMinimumHeight(190)
        self.steps_table.verticalHeader().setVisible(False)
        self._configure_table(self.steps_table, min_row_height=31)
        steps_layout.addWidget(self.steps_table)
        table_layout.addWidget(steps_group)

        plot_tab = QWidget()
        plot_tab_layout = QVBoxLayout(plot_tab)
        plot_tab_layout.setContentsMargins(0, 0, 0, 0)
        plot_tab_layout.setSpacing(12)

        plot_group = QGroupBox("Графики")
        plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout = QVBoxLayout(plot_group)
        self.formula_label = QLabel("Формула: —")
        self.formula_label.setWordWrap(True)
        self.formula_label.setStyleSheet("font-size: 13px; color: #c8d6ef;")

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(QLabel("Режим:"))
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItem("Контуры 2D", "contour")
        self.plot_mode_combo.addItem("3D поверхность", "surface")
        self.plot_mode_combo.currentIndexChanged.connect(self._on_plot_mode_changed)
        mode_layout.addWidget(self.plot_mode_combo)
        mode_layout.addStretch(1)

        self.canvas = MplCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(620)
        plot_layout.addWidget(self.formula_label)
        plot_layout.addWidget(mode_row)
        plot_layout.addWidget(self.canvas)
        plot_tab_layout.addWidget(plot_group)

        tabs.addTab(table_tab, "Таблицы")
        tabs.addTab(plot_tab, "Графики")
        return panel

    @staticmethod
    def _configure_table(table: QTableWidget, min_row_height: int) -> None:
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setTextElideMode(Qt.ElideRight)
        table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        table.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
        table.verticalHeader().setDefaultSectionSize(min_row_height)
        table.verticalHeader().setMinimumSectionSize(min_row_height)

    def _on_preset_selected(self, preset_key: str, checked: bool) -> None:
        if not checked:
            return
        self._apply_preset(preset_key)
        for key, button in self.preset_buttons.items():
            button.setProperty("checked", "true" if key == preset_key else "false")
            button.style().unpolish(button)
            button.style().polish(button)

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
                row.append(parse_float(raw, f"c[{i}][{j}]"))
            matrix.append(tuple(row))
        return tuple(matrix)

    def _run_clicked(self) -> None:
        try:
            matrix = self._read_coefficient_matrix()
            polynomial = build_polynomial(PRESET_CONFIGS[self._active_preset_key]["formula_text"], matrix)
            epsilons = parse_epsilons(self._collect_epsilons_raw())
            starts = parse_points(self._collect_start_points_raw())
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка ввода", str(exc))
            return

        try:
            batch_result, metrics = run_batch(polynomial, epsilons, starts)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка расчета", str(exc))
            return

        self._batch_result = batch_result
        self._selected_run_index = None
        self.formula_label.setText(f"Формула: {format_polynomial(batch_result.polynomial)}")
        self._fill_summary_table(batch_result)
        self.steps_table.setRowCount(0)
        self._set_steps_table_empty_layout()
        self._clear_plot()

        if batch_result.runs:
            self.summary_table.selectRow(0)

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
                "OK" if run.success else "MAX_ITER",
            ]
            for col, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 6:
                    if run.success:
                        item.setForeground(Qt.GlobalColor.green)
                self.summary_table.setItem(index, col, item)
        if batch_result.runs:
            self._set_summary_table_data_layout()
        else:
            self._set_summary_table_empty_layout()

    def _add_epsilon_input(self, value: str = "") -> None:
        item = QWidget()
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(ROW_CONTROL_SPACING)

        epsilon_input = QLineEdit(str(value))
        epsilon_input.setProperty("role", "epsilon-item")
        epsilon_input.setPlaceholderText("ε")
        epsilon_input.setFixedWidth(EPSILON_INPUT_WIDTH)
        epsilon_input.setFixedHeight(CONTROL_BUTTON_SIZE)
        epsilon_input.setAlignment(Qt.AlignCenter)
        remove_button = QPushButton("−")
        remove_button.setProperty("role", "epsilon-remove")
        remove_button.clicked.connect(lambda _checked=False, line=epsilon_input: self._remove_epsilon_input(line))

        item_layout.addWidget(epsilon_input)
        item_layout.addWidget(remove_button)
        item.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        item.setFixedWidth(EPSILON_INPUT_WIDTH + CONTROL_BUTTON_SIZE + ROW_CONTROL_SPACING)
        item.setFixedHeight(CONTROL_BUTTON_SIZE)
        self.epsilon_fields_layout.addWidget(item)
        self.epsilon_items.append((item, epsilon_input, remove_button))
        self._update_epsilon_remove_buttons()
        self._refresh_epsilon_container_width()
        epsilon_scroll = self.epsilon_scroll.horizontalScrollBar()
        if len(self.epsilon_items) > 1:
            epsilon_scroll.setValue(epsilon_scroll.maximum())
        else:
            epsilon_scroll.setValue(0)

    def _remove_epsilon_input(self, target: QLineEdit) -> None:
        if len(self.epsilon_items) <= 1:
            target.clear()
            return

        for idx, (item_widget, line_edit, _remove_button) in enumerate(self.epsilon_items):
            if line_edit is not target:
                continue
            self.epsilon_items.pop(idx)
            self.epsilon_fields_layout.removeWidget(item_widget)
            item_widget.deleteLater()
            break

        self._update_epsilon_remove_buttons()
        self._refresh_epsilon_container_width()

    def _update_epsilon_remove_buttons(self) -> None:
        disable_remove = len(self.epsilon_items) <= 1
        for _item_widget, _line_edit, remove_button in self.epsilon_items:
            remove_button.setDisabled(disable_remove)

    def _refresh_epsilon_container_width(self) -> None:
        self.epsilon_fields_container.adjustSize()
        width = self.epsilon_fields_container.sizeHint().width()
        height = self.epsilon_fields_container.sizeHint().height()
        self.epsilon_fields_container.resize(width, max(height, 56))

    def _collect_epsilons_raw(self) -> str:
        values = [field.text().strip() for _widget, field, _button in self.epsilon_items if field.text().strip()]
        if not values:
            raise ValueError("Список epsilon пуст.")
        return ",".join(values)

    def _add_start_point_input(self, x1_value: str = "", x2_value: str = "") -> None:
        item = QWidget()
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(ROW_CONTROL_SPACING)

        x1_input = QLineEdit(str(x1_value))
        x1_input.setProperty("role", "start-item")
        x1_input.setPlaceholderText("x1")
        x1_input.setFixedWidth(START_INPUT_WIDTH)
        x1_input.setFixedHeight(CONTROL_BUTTON_SIZE)
        x1_input.setAlignment(Qt.AlignCenter)

        x2_input = QLineEdit(str(x2_value))
        x2_input.setProperty("role", "start-item")
        x2_input.setPlaceholderText("x2")
        x2_input.setFixedWidth(START_INPUT_WIDTH)
        x2_input.setFixedHeight(CONTROL_BUTTON_SIZE)
        x2_input.setAlignment(Qt.AlignCenter)

        separator = QLabel("—")
        separator.setProperty("role", "start-separator")
        separator.setAlignment(Qt.AlignCenter)
        separator.setFixedWidth(START_SEPARATOR_WIDTH)

        remove_button = QPushButton("−")
        remove_button.setProperty("role", "start-remove")
        remove_button.clicked.connect(lambda _checked=False, target=x1_input: self._remove_start_point_input(target))

        item_layout.addWidget(x1_input)
        item_layout.addWidget(separator)
        item_layout.addWidget(x2_input)
        item_layout.addWidget(remove_button)
        item.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        item.setFixedWidth(
            START_INPUT_WIDTH
            + START_SEPARATOR_WIDTH
            + START_INPUT_WIDTH
            + CONTROL_BUTTON_SIZE
            + ROW_CONTROL_SPACING * 3
        )
        item.setFixedHeight(CONTROL_BUTTON_SIZE)

        self.start_points_layout.addWidget(item)
        self.start_point_items.append((item, x1_input, x2_input, remove_button))
        self._update_start_point_remove_buttons()
        self._refresh_start_points_container_width()
        start_scroll = self.start_points_scroll.horizontalScrollBar()
        if len(self.start_point_items) > 1:
            start_scroll.setValue(start_scroll.maximum())
        else:
            start_scroll.setValue(0)

    def _remove_start_point_input(self, target_x1: QLineEdit) -> None:
        if len(self.start_point_items) <= 1:
            for _item_widget, x1_input, x2_input, _button in self.start_point_items:
                if x1_input is target_x1:
                    x1_input.clear()
                    x2_input.clear()
                    break
            return

        for idx, (item_widget, x1_input, _x2_input, _remove_button) in enumerate(self.start_point_items):
            if x1_input is not target_x1:
                continue
            self.start_point_items.pop(idx)
            self.start_points_layout.removeWidget(item_widget)
            item_widget.deleteLater()
            break

        self._update_start_point_remove_buttons()
        self._refresh_start_points_container_width()

    def _update_start_point_remove_buttons(self) -> None:
        disable_remove = len(self.start_point_items) <= 1
        for _item_widget, _x1_input, _x2_input, remove_button in self.start_point_items:
            remove_button.setDisabled(disable_remove)

    def _refresh_start_points_container_width(self) -> None:
        self.start_points_container.adjustSize()
        width = self.start_points_container.sizeHint().width()
        height = self.start_points_container.sizeHint().height()
        self.start_points_container.resize(width, max(height, 56))

    def _collect_start_points_raw(self) -> str:
        points: list[str] = []
        for _item_widget, x1_input, x2_input, _remove_button in self.start_point_items:
            x1_value = x1_input.text().strip()
            x2_value = x2_input.text().strip()
            if not x1_value and not x2_value:
                continue
            if not x1_value or not x2_value:
                raise ValueError("Каждая стартовая точка должна содержать и x1, и x2.")
            points.append(f"{x1_value};{x2_value}")
        if not points:
            raise ValueError("Список стартовых точек пуст.")
        return " | ".join(points)

    def _set_start_points_raw(self, raw: str) -> None:
        for item_widget, _x1, _x2, _button in self.start_point_items:
            self.start_points_layout.removeWidget(item_widget)
            item_widget.deleteLater()
        self.start_point_items.clear()

        chunks = [chunk.strip() for chunk in raw.split("|") if chunk.strip()]
        if not chunks:
            self._add_start_point_input()
            return
        for chunk in chunks:
            parts = [part.strip() for part in chunk.split(";")]
            if len(parts) != 2:
                continue
            self._add_start_point_input(parts[0], parts[1])
        if not self.start_point_items:
            self._add_start_point_input()
        self.start_points_scroll.horizontalScrollBar().setValue(0)

    def _on_summary_selection_changed(self) -> None:
        if not self._batch_result:
            return
        selected_indexes = self.summary_table.selectionModel().selectedRows()
        if not selected_indexes:
            return
        run_index = selected_indexes[0].row()
        if run_index < 0 or run_index >= len(self._batch_result.runs):
            return
        self._selected_run_index = run_index
        run = self._batch_result.runs[run_index]
        self._fill_steps_table(run)
        self._draw_run_plot(self._batch_result, run)

    def _on_plot_mode_changed(self) -> None:
        if not self._batch_result or self._selected_run_index is None:
            self._clear_plot()
            return
        if self._selected_run_index < 0 or self._selected_run_index >= len(self._batch_result.runs):
            self._clear_plot()
            return
        run = self._batch_result.runs[self._selected_run_index]
        self._draw_run_plot(self._batch_result, run)

    def _set_summary_table_empty_layout(self) -> None:
        """Пустая таблица итогов должна занимать ширину равномерно."""
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _set_summary_table_data_layout(self) -> None:
        """Для данных: ширина по содержимому + горизонтальный скролл."""
        header = self.summary_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.summary_table.resizeColumnsToContents()
        min_widths = [48, 76, 140, 140, 110, 58, 120]
        for column, min_width in enumerate(min_widths):
            self.summary_table.setColumnWidth(column, max(self.summary_table.columnWidth(column), min_width))

    def _set_steps_table_empty_layout(self) -> None:
        """Для пустого состояния убираем визуальный «обрубок» справа."""
        self.steps_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _set_steps_table_data_layout(self) -> None:
        """Для данных: ширина по содержимому + горизонтальный скролл."""
        header = self.steps_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.steps_table.resizeColumnsToContents()
        min_widths = [52, 112, 92, 44, 112, 112, 92, 92, 120, 110]
        for column, min_width in enumerate(min_widths):
            self.steps_table.setColumnWidth(column, max(self.steps_table.columnWidth(column), min_width))

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

        with mpl.rc_context(MATPLOTLIB_DARK_RC):
            self.canvas.figure.clear()
            self.canvas.figure.patch.set_facecolor("#171b24")
            mode = str(self.plot_mode_combo.currentData())
            if mode == "surface":
                ax_surface = self.canvas.figure.add_subplot(1, 1, 1, projection="3d")
                z_clipped = np.clip(mesh_z, np.nanpercentile(mesh_z, 5), np.nanpercentile(mesh_z, 95))
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
                ax_surface.xaxis.pane.set_facecolor((0.12, 0.16, 0.23, 0.45))
                ax_surface.yaxis.pane.set_facecolor((0.12, 0.16, 0.23, 0.45))
                ax_surface.zaxis.pane.set_facecolor((0.12, 0.16, 0.23, 0.45))
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

    def _clear_plot(self) -> None:
        self.canvas.figure.clear()
        self.canvas.figure.patch.set_facecolor("#171b24")
        ax = self.canvas.figure.add_subplot(1, 1, 1)
        ax.set_facecolor("#10141f")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#324866")
        ax.text(
            0.5,
            0.5,
            "Выберите запуск из таблицы, чтобы увидеть графики",
            ha="center",
            va="center",
            color="#c8cfdb",
            fontsize=12,
        )
        self.canvas.draw_idle()

    @staticmethod
    def _format_point(point: tuple[float, ...]) -> str:
        return "(" + ", ".join(f"{value:.6g}" for value in point) + ")"


def main() -> None:
    """Запускает Qt-приложение."""
    app = QApplication(sys.argv)
    window = RosenbrockWindow()
    window.show()
    sys.exit(app.exec())
