"""Общие Qt-утилиты для UI-слоя лабораторных."""

from optim_core.ui.controls_builder import (
    ControlsPanel,
    add_parameter_row,
    create_choice_chip_grid,
    create_parameter_grid,
    create_controls_panel,
    create_flush_row,
    create_primary_action_button,
    create_standard_group,
)
from optim_core.ui.dynamic_inputs import DynamicSeriesInputRow
from optim_core.ui.empty_state import EmptyStateStack, create_centered_empty_state_page
from optim_core.ui.math_header import MathHeaderView
from optim_core.ui.plot_canvas import (
    DARK_MATPLOTLIB_RC,
    PlotCanvas,
    clear_plot_canvas,
    dark_plot_context,
)
from optim_core.ui.qt_layout import (
    configure_two_panel_splitter,
    create_scroll_container,
)
from optim_core.ui.results_workspace import ResultsWorkspace, create_results_workspace
from optim_core.ui.run_flow import BatchRunUiController, BatchRunUiHooks
from optim_core.ui.tab_layout import ResultsPlotTabIndexes, configure_results_plot_tabs
from optim_core.ui.table_widgets import (
    configure_data_table,
    set_table_data_layout,
    set_table_empty_layout,
)
from optim_core.ui.task_controller import TaskController
from optim_core.ui.theme import (
    DarkQtThemeTokens,
    build_dark_qt_base_styles,
    build_choice_chip_styles,
    build_dynamic_series_styles,
)

__all__ = (
    "ControlsPanel",
    "create_parameter_grid",
    "add_parameter_row",
    "create_choice_chip_grid",
    "create_controls_panel",
    "create_standard_group",
    "create_primary_action_button",
    "create_flush_row",
    "DynamicSeriesInputRow",
    "create_centered_empty_state_page",
    "EmptyStateStack",
    "MathHeaderView",
    "PlotCanvas",
    "DARK_MATPLOTLIB_RC",
    "clear_plot_canvas",
    "dark_plot_context",
    "ResultsWorkspace",
    "create_results_workspace",
    "create_scroll_container",
    "configure_two_panel_splitter",
    "BatchRunUiController",
    "BatchRunUiHooks",
    "TaskController",
    "ResultsPlotTabIndexes",
    "configure_results_plot_tabs",
    "configure_data_table",
    "set_table_empty_layout",
    "set_table_data_layout",
    "DarkQtThemeTokens",
    "build_dark_qt_base_styles",
    "build_choice_chip_styles",
    "build_dynamic_series_styles",
)
