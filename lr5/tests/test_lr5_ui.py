"""UI-smoke тесты для ЛР5."""

from __future__ import annotations

from lr2.domain.models import SolverResult
from lr5.application.services import build_method_config, variant_2_problem
from lr5.domain.models import BarrierIterationResult, BarrierResult
from lr5.ui.window import BarrierWindow


def _wait_for_task(app, window: BarrierWindow) -> None:
    for _ in range(200):
        app.processEvents()
        if not window._run_task.is_running():
            return
    raise AssertionError("Фоновая задача ЛР5 не завершилась вовремя")


def _make_iteration(
    *,
    problem,
    k: int,
    mu_k: float,
    start_point: tuple[float, float],
    point: tuple[float, float],
    inner_stop_reason: str = "Достигнут критерий ||x_(k+1) - x_k|| <= epsilon",
) -> BarrierIterationResult:
    objective_value = problem.objective(point)
    constraints_values = tuple(constraint.evaluator(point) for constraint in problem.constraints)
    inner_result = SolverResult(
        epsilon=1e-4,
        start_point=start_point,
        optimum_point=point,
        optimum_value=objective_value,
        iterations_count=3,
        steps=(),
        trajectory=(start_point, point),
        success=True,
        stop_reason=inner_stop_reason,
    )
    barrier_value = 1.0
    barrier_metric = 0.5
    barrier_metric_term = mu_k * barrier_metric
    barrier_term = mu_k * barrier_value
    return BarrierIterationResult(
        k=k,
        mu_k=mu_k,
        start_point=start_point,
        x_mu_k=point,
        objective_value=objective_value,
        barrier_value=barrier_value,
        barrier_metric=barrier_metric,
        barrier_metric_term=barrier_metric_term,
        theta_value=objective_value + barrier_term,
        barrier_term=barrier_term,
        constraints_values=constraints_values,
        inner_result=inner_result,
    )


def test_lr5_window_renders_controls_and_results(qapp_offscreen) -> None:
    app = qapp_offscreen
    window = BarrierWindow()

    assert window.windowTitle() == "ЛР5 — Метод барьерных функций"
    assert window.start_x1_input.text() == "0"
    assert window.start_x2_input.text() == "0"
    assert window.mu0_input.text() == "10"
    assert window.beta_input.text() == "0.1"
    assert window.outer_epsilon_input.text() == "1e-3"
    assert window.outer_max_iterations_input.text() == "20"
    assert window.barrier_buttons["reciprocal"].isChecked()
    assert window.save_button.text() == "Сохранить результаты"
    assert window.advanced_toggle.text() == "Дополнительные"
    assert window.advanced_toggle.isChecked() is False
    assert window.inner_details_toggle.text() == "Подробности внутренних итераций"
    assert window.inner_details_toggle.isChecked() is False
    assert window.save_button.isEnabled() is False
    assert window.report_empty_stack.currentIndex() == 0
    assert window.plot_empty_stack.currentIndex() == 0

    window._on_run_clicked()
    _wait_for_task(app, window)
    app.processEvents()

    assert window.outer_table.columnCount() == 10
    assert window.outer_table.rowCount() >= 1
    assert window.outer_table.item(0, 1).text() == "10"
    assert window.inner_table.rowCount() > 0
    assert window.canvas.figure.axes
    assert window.summary_status.text() == "Готово"
    assert window.summary_comment.text().startswith("Внешний цикл завершён")
    assert window.summary_point_caption.text() == "Итоговая точка"
    assert window.summary_point.text() != "—"
    assert window.summary_constraints.text().startswith("g1(x) = ")
    assert window.summary_conclusion.text() == "Получено корректное допустимое решение."
    assert "Готово" in window.status_label.text()
    assert window.report_empty_stack.currentIndex() == 1
    assert window.plot_empty_stack.currentIndex() == 1


def test_lr5_window_renders_warning_and_error_states(qapp_offscreen) -> None:
    _ = qapp_offscreen
    problem = variant_2_problem()
    config = build_method_config()
    valid_iteration = _make_iteration(
        problem=problem,
        k=1,
        mu_k=config.mu0,
        start_point=(0.0, 0.0),
        point=(2.0, 0.5),
    )
    failed_iteration = _make_iteration(
        problem=problem,
        k=2,
        mu_k=1.0,
        start_point=valid_iteration.x_mu_k,
        point=(3.2, 0.8),
        inner_stop_reason="Следующий шаг нарушил допустимость",
    )

    window = BarrierWindow()
    warning_result = BarrierResult(
        problem=problem,
        start_point=(0.0, 0.0),
        config=config,
        iterations=(valid_iteration,),
        last_valid_outer_iteration=valid_iteration,
        failed_outer_iteration=failed_iteration,
        optimum_point=valid_iteration.x_mu_k,
        optimum_value=valid_iteration.objective_value,
        optimum_constraints=valid_iteration.constraints_values,
        success=False,
        stop_reason="Расчёт остановлен с предупреждением: следующий шаг нарушил допустимость или численный контракт",
        status="warning",
    )
    window._on_run_succeeded(warning_result)

    assert window.summary_status.text() == "Расчёт остановлен с предупреждением"
    assert window.summary_point_caption.text() == "Последняя допустимая точка"
    assert window.summary_point.text() == window._format_point(valid_iteration.x_mu_k)
    assert window.summary_conclusion.text() == "Показана последняя допустимая точка; следующий шаг оказался недопустимым."
    assert window.outer_table.rowCount() == 1
    assert all("inf" not in window.outer_table.item(0, col).text() for col in range(window.outer_table.columnCount()))
    assert "Предупреждение" in window.status_label.text()

    error_result = BarrierResult(
        problem=problem,
        start_point=(0.0, 0.0),
        config=config,
        iterations=(),
        last_valid_outer_iteration=None,
        failed_outer_iteration=failed_iteration,
        optimum_point=(0.0, 0.0),
        optimum_value=problem.objective((0.0, 0.0)),
        optimum_constraints=tuple(constraint.evaluator((0.0, 0.0)) for constraint in problem.constraints),
        success=False,
        stop_reason="Расчёт завершён без валидного допустимого результата: следующий шаг нарушил допустимость или численный контракт",
        status="error",
    )
    window._on_run_succeeded(error_result)

    assert window.summary_status.text() == "Ошибка"
    assert window.summary_point.text() == "—"
    assert window.summary_conclusion.text() == "Валидное допустимое решение не найдено."
    assert window.outer_table.rowCount() == 0
    assert "без валидного" in window.status_label.text()


def test_lr5_window_rejects_invalid_start_point(qapp_offscreen) -> None:
    _ = qapp_offscreen
    window = BarrierWindow()

    window.start_x1_input.setText("3")
    window.start_x2_input.setText("1")
    window._on_run_clicked()

    assert "строго внутренней" in window.status_label.text()
    assert window.outer_table.rowCount() == 0
