"""Smoke-тесты корректности методов ЛР3."""

from __future__ import annotations

from lr3.application.services import build_config, run_conjugate, run_gradient


def test_gradient_method_improves_quadratic_objective() -> None:
    config = build_config(
        epsilon_raw="1e-5",
        max_iterations_raw="250",
        initial_step_raw="0.1",
        timeout_raw="2.0",
    )
    expression = "-(x1-1)**2 - (x2+2)**2"
    result, _ = run_gradient(expression=expression, start_point=(7.0, -9.0), config=config)

    assert result.optimum_value > -0.05
    assert result.iterations_count > 0


def test_conjugate_method_moves_to_quadratic_peak() -> None:
    config = build_config(
        epsilon_raw="1e-5",
        max_iterations_raw="250",
        initial_step_raw="0.1",
        timeout_raw="2.0",
    )
    expression = "-(x1-3)**2 - 2*(x2-4)**2"
    result, _ = run_conjugate(expression=expression, start_point=(0.0, 0.0), config=config)

    assert result.optimum_value > -0.1
    assert abs(result.optimum_point[0] - 3.0) < 0.5
    assert abs(result.optimum_point[1] - 4.0) < 0.5
