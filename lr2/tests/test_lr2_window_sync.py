"""UI-smoke тесты синхронизации результатов ЛР2."""

from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication

from lr2.application.services import VARIANT_PRESETS, build_polynomial, run_batch
from lr2.domain.models import BatchResult, SolverResult
from lr2.ui.window import RosenbrockWindow


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.parametrize(
    "start_point",
    (
        (0.75, 0.0),
        (1.0, -1.25),
    ),
    ids=("start_0_75_0", "start_1_-1_25"),
)
def test_lr2_window_syncs_summary_iterations_and_plot(start_point: tuple[float, float]) -> None:
    app = _app()
    window = RosenbrockWindow()
    polynomial = build_polynomial("F1", VARIANT_PRESETS["variant_f1"])
    batch_result, _metrics = run_batch(polynomial, (0.1,), (start_point,))

    window._run_flow.apply(batch_result)
    app.processEvents()

    run = batch_result.runs[0]

    assert window.summary_table.rowCount() == 1
    assert window.summary_table.selectionModel() is not None
    assert window.summary_table.selectionModel().selectedRows()[0].row() == 0
    assert window._selected_run_index == 0
    assert window.steps_table.rowCount() == len(run.steps)
    assert window.steps_state_label.text() == "Выберите запуск, чтобы увидеть итерации."
    assert window.canvas.figure.axes
    assert window.canvas.figure.axes[0].collections

    window.results_tabs.setCurrentIndex(window.results_tab_indexes.plot)
    window._on_results_tab_changed(window.results_tab_indexes.plot)
    app.processEvents()

    assert window.canvas.figure.axes
    assert window.canvas.figure.axes[0].collections


def test_lr2_window_shows_empty_state_when_run_has_no_detail_payload() -> None:
    app = _app()
    window = RosenbrockWindow()
    batch_result = BatchResult(
        polynomial=build_polynomial("F1", VARIANT_PRESETS["variant_f1"]),
        runs=(
            SolverResult(
                epsilon=0.1,
                start_point=(0.75, 0.0),
                optimum_point=(0.75, 0.0),
                optimum_value=0.0,
                iterations_count=0,
                steps=(),
                trajectory=(),
                success=True,
                stop_reason="Нет детальных данных",
            ),
        ),
    )

    window._run_flow.apply(batch_result)
    app.processEvents()

    assert window.summary_table.rowCount() == 1
    assert window.steps_table.rowCount() == 0
    assert "нет сохранённой истории итераций" in window.steps_state_label.text()
    assert window.canvas.figure.axes
    assert any(
        "данных для графика" in text.get_text()
        for text in window.canvas.figure.axes[0].texts
    )
