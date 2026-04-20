"""Контрактные тесты ЛР5."""

from __future__ import annotations

import math

import pytest

from lr2.domain.models import SolverResult
from lr5.application.services import build_method_config, run_barrier_method, variant_2_problem
from lr5.domain.barrier import LOG_BARRIER, RECIPROCAL_BARRIER, barrier_value, is_strictly_feasible


def test_variant_2_start_point_is_strictly_feasible() -> None:
    problem = variant_2_problem()
    assert is_strictly_feasible((0.0, 0.0), problem.constraints)


@pytest.mark.parametrize("barrier_kind", (RECIPROCAL_BARRIER, LOG_BARRIER))
def test_barrier_method_converges_on_variant_2(barrier_kind: str) -> None:
    problem = variant_2_problem()
    config = build_method_config(barrier_kind=barrier_kind)
    result = run_barrier_method(problem, (0.0, 0.0), config)

    assert result.success
    assert result.status == "success"
    assert result.iterations
    assert result.last_valid_outer_iteration is not None
    assert result.failed_outer_iteration is None
    assert result.iterations[0].mu_k == pytest.approx(10.0)
    assert result.iterations[0].barrier_metric > 0.0
    assert result.optimum_point == pytest.approx((2.5, 0.5), abs=0.2)
    assert result.optimum_value == pytest.approx(12.5, abs=0.25)
    assert result.optimum_constraints[1] == pytest.approx(0.0, abs=0.2)
    assert result.optimum_constraints[0] < 0.0
    assert result.stop_reason


def test_barrier_rejects_infeasible_start_point() -> None:
    problem = variant_2_problem()
    config = build_method_config()

    with pytest.raises(ValueError, match="Стартовая точка должна быть строго допустимой"):
        run_barrier_method(problem, (3.0, 1.0), config)


def test_barrier_value_returns_infinity_outside_domain() -> None:
    problem = variant_2_problem()
    assert math.isinf(barrier_value((3.0, 1.0), problem.constraints, RECIPROCAL_BARRIER))


def test_outer_mu_decreases_geometrically_and_stops_by_max_iterations() -> None:
    problem = variant_2_problem()
    config = build_method_config(mu0=10.0, beta=0.1, epsilon_outer=1e-300, max_outer_iterations=3)
    result = run_barrier_method(problem, (0.0, 0.0), config)

    assert [iteration.mu_k for iteration in result.iterations] == pytest.approx([10.0, 1.0, 0.1])
    assert result.status == "warning"
    assert result.last_valid_outer_iteration is not None
    assert result.stop_reason == "Внешний цикл остановлен по защитному лимиту внешних итераций (3)"


def test_outer_cycle_stops_by_outer_metric() -> None:
    problem = variant_2_problem()
    config = build_method_config(mu0=10.0, beta=0.1, epsilon_outer=1e9, max_outer_iterations=10)
    result = run_barrier_method(problem, (0.0, 0.0), config)

    assert len(result.iterations) == 1
    assert result.status == "success"
    assert result.iterations[0].barrier_metric_term < config.epsilon_outer
    assert "критерию mu_k * M(x_mu_k)" in result.stop_reason


def test_barrier_method_returns_warning_without_promoting_invalid_iteration(monkeypatch) -> None:
    problem = variant_2_problem()
    config = build_method_config(mu0=10.0, beta=0.1, epsilon_outer=1e-300, max_outer_iterations=4)
    calls = {"count": 0}

    def fake_rosenbrock_minimize(*, objective, start_point, config):  # noqa: ANN001
        _ = objective, config
        calls["count"] += 1
        if calls["count"] == 1:
            return SolverResult(
                epsilon=1e-4,
                start_point=start_point,
                optimum_point=(2.0, 0.5),
                optimum_value=13.25,
                iterations_count=1,
                steps=(),
                trajectory=(start_point, (2.0, 0.5)),
                success=True,
                stop_reason="ok",
            )
        return SolverResult(
            epsilon=1e-4,
            start_point=start_point,
            optimum_point=(3.2, 0.8),
            optimum_value=0.0,
            iterations_count=1,
            steps=(),
            trajectory=(start_point, (3.2, 0.8)),
            success=True,
            stop_reason="ok",
        )

    monkeypatch.setattr("lr5.application.services.rosenbrock_minimize", fake_rosenbrock_minimize)

    result = run_barrier_method(problem, (0.0, 0.0), config)

    assert result.status == "warning"
    assert not result.success
    assert len(result.iterations) == 1
    assert result.last_valid_outer_iteration == result.iterations[0]
    assert result.failed_outer_iteration is not None
    assert result.failed_outer_iteration.x_mu_k == pytest.approx((3.2, 0.8))
    assert "нарушил допустимость" in result.stop_reason.lower()


def test_barrier_method_returns_error_if_first_outer_step_is_invalid(monkeypatch) -> None:
    problem = variant_2_problem()
    config = build_method_config(mu0=10.0, beta=0.1, epsilon_outer=1e-300, max_outer_iterations=4)

    def fake_rosenbrock_minimize(*, objective, start_point, config):  # noqa: ANN001
        _ = objective, config
        return SolverResult(
            epsilon=1e-4,
            start_point=start_point,
            optimum_point=(3.2, 0.8),
            optimum_value=0.0,
            iterations_count=1,
            steps=(),
            trajectory=(start_point, (3.2, 0.8)),
            success=True,
            stop_reason="ok",
        )

    monkeypatch.setattr("lr5.application.services.rosenbrock_minimize", fake_rosenbrock_minimize)

    result = run_barrier_method(problem, (0.0, 0.0), config)

    assert result.status == "error"
    assert not result.success
    assert result.iterations == ()
    assert result.last_valid_outer_iteration is None
    assert result.failed_outer_iteration is not None
    assert "без валидного допустимого результата" in result.stop_reason


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mu0": 0.0}, "mu0"),
        ({"beta": 1.0}, "beta"),
        ({"beta": 0.0}, "beta"),
        ({"epsilon_outer": 0.0}, "epsilon"),
        ({"max_outer_iterations": 0}, "max_outer_iterations"),
    ],
)
def test_build_method_config_validates_outer_parameters(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_method_config(**kwargs)
