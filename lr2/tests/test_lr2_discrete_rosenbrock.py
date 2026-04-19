"""Контрактные тесты дискретного варианта метода Розенброка."""

from __future__ import annotations

import pytest
from lr2.application.services import (
    DEFAULT_DISCRETE_SOLVER_CONFIG,
    build_polynomial,
    run_discrete_batch,
)
from lr2.domain.models import DiscreteSolverConfig, Vector
from lr2.domain.rosenbrock import discrete_rosenbrock_minimize


def _config(epsilon: float, *, delta_step: float = 0.2) -> DiscreteSolverConfig:
    return DiscreteSolverConfig(
        epsilon=epsilon,
        max_iterations=int(DEFAULT_DISCRETE_SOLVER_CONFIG["max_iterations"]),
        delta_step=delta_step,
        alpha=float(DEFAULT_DISCRETE_SOLVER_CONFIG["alpha"]),
        beta=float(DEFAULT_DISCRETE_SOLVER_CONFIG["beta"]),
        direction_zero_tolerance=float(DEFAULT_DISCRETE_SOLVER_CONFIG["direction_zero_tolerance"]),
    )


def test_discrete_rosenbrock_converges_on_quadratic() -> None:
    target = (1.0, -2.0)

    def objective(vector: Vector) -> float:
        return (vector[0] - target[0]) ** 2 + (vector[1] - target[1]) ** 2

    result = discrete_rosenbrock_minimize(
        objective=objective,
        start_point=(4.0, -3.0),
        config=_config(1e-3),
    )

    assert result.success
    assert result.optimum_point == pytest.approx(target, abs=8e-2)
    assert result.optimum_value == pytest.approx(0.0, abs=1e-3)
    assert result.steps
    assert result.trajectory[0] == pytest.approx((4.0, -3.0))


def test_discrete_rosenbrock_rejects_nonpositive_delta_step() -> None:
    with pytest.raises(ValueError, match="delta_step должен быть > 0"):
        discrete_rosenbrock_minimize(
            objective=lambda vector: sum(value * value for value in vector),
            start_point=(1.0, 1.0),
            config=_config(1e-3, delta_step=0.0),
        )


def test_run_discrete_batch_wraps_single_run() -> None:
    polynomial = build_polynomial(
        "Quadratic",
        (
            (5.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
        ),
    )

    batch_result, metrics = run_discrete_batch(
        polynomial=polynomial,
        epsilons=(1e-3,),
        start_points=((4.0, -3.0),),
    )

    assert metrics.total_count == 1
    assert metrics.run_count == 1
    assert metrics.failure_count == 0
    assert metrics.domain_refusal_count == 0
    assert metrics.unexpected_error_count == 0
    assert batch_result.runs
    assert batch_result.runs[0].success
    assert batch_result.summary.total_count == 1
    assert batch_result.summary.success_count == 1
    assert batch_result.summary.domain_refusal_count == 0
    assert batch_result.summary.unexpected_error_count == 0
