"""Переиспользуемые генераторы Qt-стилей."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DarkQtThemeTokens:
    background: str
    text: str
    font_family: str
    group_border: str
    group_radius_px: int
    group_padding_px: int
    group_title_color: str
    button_bg: str
    button_border: str
    button_hover_bg: str
    button_pressed_bg: str
    button_disabled_bg: str
    button_disabled_text: str
    button_disabled_border: str
    primary_bg: str
    primary_border: str
    primary_hover_bg: str
    primary_pressed_bg: str
    tab_bg: str
    tab_border: str
    tab_selected_bg: str
    tab_selected_border: str


def build_dark_qt_base_styles(tokens: DarkQtThemeTokens) -> str:
    """Строит общий базовый QSS для темной темы Qt."""
    return f"""
            QMainWindow, QWidget {{
                background: {tokens.background};
                color: {tokens.text};
                font-family: {tokens.font_family};
                font-size: 15px;
            }}
            QLabel {{
                background: transparent;
            }}
            QLabel#SectionCaption {{
                color: #9aa5bb;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}
            QLabel[role="parameter-label"] {{
                color: #dce6f5;
                font-size: 15px;
                font-weight: 600;
            }}
            QLabel[role="formula-preview"] {{
                background: #101827;
                border: 1px solid #304665;
                border-radius: 12px;
                padding: 10px 14px;
                color: #ecf3ff;
                font-size: 18px;
                font-weight: 700;
            }}
            QLabel#SectionHint {{
                color: #a8b1c3;
                font-size: 12px;
            }}
            QLabel#SummaryEmptyTitle {{
                color: #eef2f8;
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#SummaryEmptyText {{
                color: #b8c1d1;
                font-size: 15px;
            }}
            QWidget#SummaryEmptyCard {{
                background: #181b24;
                border: 1px solid #31384a;
                border-radius: 14px;
            }}
            QLineEdit {{
                background: #131824;
                border: 1px solid #3f4a62;
                border-radius: 8px;
                padding: 7px 10px;
                color: #f5f7fb;
                selection-background-color: #2379ff;
            }}
            QLineEdit:focus {{
                border: 1px solid #2f8fff;
            }}
            QTableWidget {{
                background: #12161d;
                border: 1px solid #464b59;
                border-radius: 8px;
                color: #f5f7fb;
                gridline-color: #2d3241;
                selection-background-color: #2a6df4;
                alternate-background-color: #151b25;
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
                font-size: 14px;
            }}
            QHeaderView::section {{
                background: #222938;
                color: #dbe2ee;
                border-right: 1px solid #33415b;
                border-bottom: 1px solid #33415b;
                padding: 6px 8px;
                font-size: 12px;
                font-weight: 700;
            }}
            QTableCornerButton::section {{
                background: #222938;
                border-right: 1px solid #33415b;
                border-bottom: 1px solid #33415b;
            }}
            QGroupBox {{
                border: 1px solid {tokens.group_border};
                border-radius: {tokens.group_radius_px}px;
                margin-top: 12px;
                padding: {tokens.group_padding_px}px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: {tokens.group_title_color};
            }}
            QGroupBox#ReportCard {{
                background: #171c26;
                border: 1px solid #36425a;
            }}
            QGroupBox#IterationCard {{
                background: #141923;
                border: 1px solid #40506b;
            }}
            QLabel[role="report-caption"] {{
                color: #9ca8bc;
                font-size: 12px;
                font-weight: 700;
            }}
            QLabel[role="report-value"] {{
                color: #f2f5fa;
                font-size: 14px;
            }}
            QPushButton {{
                background: {tokens.button_bg};
                border: 1px solid {tokens.button_border};
                border-radius: 8px;
                padding: 9px 14px;
                color: #f5f7fb;
                font-weight: 600;
                min-height: 22px;
            }}
            QPushButton:hover {{
                background: {tokens.button_hover_bg};
            }}
            QPushButton:pressed {{
                background: {tokens.button_pressed_bg};
            }}
            QPushButton:disabled {{
                background: {tokens.button_disabled_bg};
                color: {tokens.button_disabled_text};
                border-color: {tokens.button_disabled_border};
            }}
            QPushButton[variant="primary"] {{
                background: {tokens.primary_bg};
                border-color: {tokens.primary_border};
            }}
            QPushButton[variant="primary"]:hover {{
                background: {tokens.primary_hover_bg};
            }}
            QPushButton[variant="primary"]:pressed {{
                background: {tokens.primary_pressed_bg};
            }}
            QTabWidget::pane {{
                border: 0;
                top: 0;
            }}
            QTabWidget::tab-bar {{
                alignment: left;
            }}
            QTabBar {{
                qproperty-drawBase: 0;
                qproperty-expanding: 1;
            }}
            QTabBar::tab {{
                background: {tokens.tab_bg};
                border: 1px solid {tokens.tab_border};
                padding: 8px 16px 10px 16px;
                margin-right: 6px;
                margin-bottom: 2px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                min-width: 0px;
                min-height: 24px;
                font-size: 14px;
                font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: {tokens.tab_selected_bg};
                border-color: {tokens.tab_selected_border};
                color: #ffffff;
            }}
    """


def build_dynamic_series_styles(
    *,
    add_role: str = "series-add",
    remove_role: str = "series-remove",
    field_role: str = "series-item",
    separator_role: str | None = None,
) -> str:
    """Возвращает QSS для динамического ряда инпутов `+/-`."""
    separator_selector = ""
    if separator_role is not None:
        separator_selector = f"""
            QLabel[role="{separator_role}"] {{
                color: #a9b8d4;
                font-size: 18px;
                font-weight: 700;
            }}
        """
    return f"""
            QPushButton[role="{add_role}"] {{
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
            }}
            QPushButton[role="{add_role}"]:hover {{ background: #2588ff; }}
            QPushButton[role="{add_role}"]:pressed {{ background: #0d66d8; }}
            QPushButton[role="{remove_role}"] {{
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
            }}
            QPushButton[role="{remove_role}"]:hover {{ background: #36415a; }}
            QPushButton[role="{remove_role}"]:pressed {{ background: #242d3d; }}
            QPushButton[role="{remove_role}"]:disabled {{
                background: #1f2533;
                color: #66738d;
                border-color: #3c465f;
            }}
            QLineEdit[role="{field_role}"] {{
                min-height: 44px;
                font-size: 14px;
                font-weight: 700;
                padding-top: 0px;
                padding-bottom: 0px;
            }}
            {separator_selector}
    """


def build_choice_chip_styles(
    *,
    role: str = "choice-chip",
) -> str:
    """Возвращает QSS для сегментированных кнопок выбора."""
    return f"""
            QPushButton[role="{role}"] {{
                min-height: 32px;
                min-width: 0px;
                padding: 8px 10px;
                border-radius: 9px;
                background: #253149;
                border: 1px solid #46567a;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton[role="{role}"]:checked {{
                background: #0f7aff;
                border-color: #3b94ff;
            }}
    """
