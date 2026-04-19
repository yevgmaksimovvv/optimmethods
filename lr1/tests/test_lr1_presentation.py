"""Тесты презентационного контракта ЛР1."""

from __future__ import annotations

from lr1.application.services import build_input_config, run_batch
from lr1.ui.tabs import build_iteration_rows, build_plot_context, build_summary_view_model


def _build_grid_report():
    config = build_input_config(
        function_key="quadratic",
        kind="max",
        method_key="dichotomy",
        a_raw="-5",
        b_raw="5",
        eps_raw="0.01",
        l_raw="0.1",
        coefficient_raws={"a": "-2", "b": "10", "c": "3"},
    )
    return run_batch(config, eps_values=(0.2, 0.01), l_values=(0.1,))


def test_lr1_presentation_helpers_cover_grid_summary_and_plot_context() -> None:
    report = _build_grid_report()
    view_model = build_summary_view_model(report)

    assert view_model.results_table.headers[0] == "Метод"
    assert len(view_model.results_table.rows) == 1
    assert view_model.show_skipped is True
    assert view_model.show_notes is True
    assert any("Пропущено" in note for note in view_model.observations)

    selected_run = report.grid_runs_by_method["dichotomy"][0]
    context_text = build_plot_context(report, "dichotomy", selected_run)
    iteration_rows = build_iteration_rows(selected_run.result)

    assert "Сверху показана вся серия" in context_text
    assert iteration_rows
    assert isinstance(iteration_rows[0].left_wins, bool)
