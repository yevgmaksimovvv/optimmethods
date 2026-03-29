"""Общие helper-функции для компоновки вкладок."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QTabWidget, QWidget


@dataclass(frozen=True)
class ResultsPlotTabIndexes:
    """Индексы стандартных вкладок 'результаты/график'."""

    results: int
    plot: int


def configure_results_plot_tabs(
    tab_widget: QTabWidget,
    *,
    results_widget: QWidget,
    plot_widget: QWidget,
    results_title: str,
    plot_title: str = "Графики",
) -> ResultsPlotTabIndexes:
    """Стандартизирует двухвкладочный layout: таблицы/итоги и графики."""
    tab_widget.setDocumentMode(True)
    tab_widget.tabBar().setExpanding(True)
    tab_widget.tabBar().setUsesScrollButtons(False)
    tab_widget.clear()
    results_index = tab_widget.addTab(results_widget, results_title)
    plot_index = tab_widget.addTab(plot_widget, plot_title)
    return ResultsPlotTabIndexes(results=results_index, plot=plot_index)
