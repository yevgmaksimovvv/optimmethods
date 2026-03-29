"""Общий каркас правой панели: вкладки `Таблицы` и `Графики`."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from optim_core.ui.empty_state import EmptyStateStack
from optim_core.ui.tab_layout import ResultsPlotTabIndexes, configure_results_plot_tabs


@dataclass(frozen=True)
class ResultsWorkspace:
    """Собранный каркас правой панели и его ключевые ссылки."""

    panel: QWidget
    tabs: QTabWidget
    tab_indexes: ResultsPlotTabIndexes
    tables_layout: QVBoxLayout
    plots_layout: QVBoxLayout
    tables_empty_stack: EmptyStateStack | None


def create_results_workspace(
    *,
    results_title: str = "Таблицы",
    plot_title: str = "Графики",
    with_tables_empty_state: bool = False,
    tables_empty_title: str = "",
    tables_empty_description: str = "",
    tables_empty_hint: str = "",
) -> ResultsWorkspace:
    """Создаёт правую панель с табами и опциональным empty-state для таблиц."""
    panel = QWidget()
    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(0, 0, 0, 0)
    panel_layout.setSpacing(12)

    tabs = QTabWidget()
    panel_layout.addWidget(tabs)

    tables_tab = QWidget()
    tables_tab_layout = QVBoxLayout(tables_tab)
    tables_tab_layout.setContentsMargins(0, 0, 0, 0)
    tables_tab_layout.setSpacing(12)

    tables_layout: QVBoxLayout
    tables_empty_stack: EmptyStateStack | None = None
    if with_tables_empty_state:
        tables_content = QWidget()
        tables_layout = QVBoxLayout(tables_content)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        tables_layout.setSpacing(12)
        tables_empty_stack = EmptyStateStack(
            title=tables_empty_title,
            description=tables_empty_description,
            hint=tables_empty_hint,
            content_widget=tables_content,
        )
        tables_tab_layout.addWidget(tables_empty_stack)
    else:
        tables_layout = tables_tab_layout

    plot_tab = QWidget()
    plot_layout = QVBoxLayout(plot_tab)
    plot_layout.setContentsMargins(0, 0, 0, 0)
    plot_layout.setSpacing(12)

    tab_indexes = configure_results_plot_tabs(
        tabs,
        results_widget=tables_tab,
        plot_widget=plot_tab,
        results_title=results_title,
        plot_title=plot_title,
    )
    return ResultsWorkspace(
        panel=panel,
        tabs=tabs,
        tab_indexes=tab_indexes,
        tables_layout=tables_layout,
        plots_layout=plot_layout,
        tables_empty_stack=tables_empty_stack,
    )
