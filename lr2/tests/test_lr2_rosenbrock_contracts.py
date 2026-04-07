"""Дополнительные контрактные тесты ядра метода Розенброка ЛР2."""

from __future__ import annotations

import pytest

from lr2.application.services import DEFAULT_DISCRETE_SOLVER_CONFIG, DEFAULT_SOLVER_CONFIG
from lr2.domain.models import DiscreteSolverConfig, SolverConfig, Vector
from lr2.domain.rosenbrock import discrete_rosenbrock_minimize, rosenbrock_minimize


def _solver_config(*, epsilon: float = 1e-3, max_iterations: int = 200) -> SolverConfig:
    return SolverConfig(
        epsilon=epsilon,
        max_iterations=max_iterations,
        line_search_min_lambda=float(DEFAULT_SOLVER_CONFIG["line_search_min_lambda"]),
        line_search_max_lambda=float(DEFAULT_SOLVER_CONFIG["line_search_max_lambda"]),
        line_search_tolerance=float(DEFAULT_SOLVER_CONFIG["line_search_tolerance"]),
        line_search_max_iterations=int(DEFAULT_SOLVER_CONFIG["line_search_max_iterations"]),
        direction_zero_tolerance=float(DEFAULT_SOLVER_CONFIG["direction_zero_tolerance"]),
        stagnation_abs_tolerance=float(DEFAULT_SOLVER_CONFIG["stagnation_abs_tolerance"]),
        stagnation_rel_tolerance=float(DEFAULT_SOLVER_CONFIG["stagnation_rel_tolerance"]),
    )


def _discrete_solver_config(*, epsilon: float = 1e-3, max_iterations: int = 200) -> DiscreteSolverConfig:
    return DiscreteSolverConfig(
        epsilon=epsilon,
        max_iterations=max_iterations,
        delta_step=float(DEFAULT_DISCRETE_SOLVER_CONFIG["delta_step"]),
        alpha=float(DEFAULT_DISCRETE_SOLVER_CONFIG["alpha"]),
        beta=float(DEFAULT_DISCRETE_SOLVER_CONFIG["beta"]),
        direction_zero_tolerance=float(DEFAULT_DISCRETE_SOLVER_CONFIG["direction_zero_tolerance"]),
    )


def test_rosenbrock_minimize_stops_by_iteration_limit() -> None:
    def objective(vector: Vector) -> float:
        return (vector[0] - 1.0) ** 2 + (vector[1] + 2.0) ** 2

    result = rosenbrock_minimize(
        objective=objective,
        start_point=(4.0, -3.0),
        config=_solver_config(epsilon=1e-12, max_iterations=1),
    )

    assert not result.success
    assert result.stop_reason == "Достигнут лимит итераций"
    assert result.iterations_count == 1
    assert len(result.steps) == 2
    assert len(result.trajectory) == 2
    assert result.optimum_value == pytest.approx(objective(result.optimum_point))


def test_discrete_rosenbrock_minimize_stops_by_delta_threshold_on_flat_objective() -> None:
    def objective(vector: Vector) -> float:
        return 0.0

    start_point = (4.0, -3.0)
    result = discrete_rosenbrock_minimize(
        objective=objective,
        start_point=start_point,
        config=_discrete_solver_config(epsilon=0.05, max_iterations=10),
    )

    assert result.success
    assert result.stop_reason == "Достигнут критерий |Δ_j| <= epsilon"
    assert result.iterations_count == 1
    assert result.optimum_point == pytest.approx(start_point)
    assert result.optimum_value == pytest.approx(0.0)
    assert len(result.steps) == 2
    assert len(result.trajectory) == 1


@pytest.mark.parametrize(
    ("config", "expected_message"),
    (
        (
            SolverConfig(
                epsilon=0.0,
                max_iterations=10,
                line_search_min_lambda=-0.5,
                line_search_max_lambda=0.5,
                line_search_tolerance=1e-5,
                line_search_max_iterations=120,
                direction_zero_tolerance=1e-12,
                stagnation_abs_tolerance=1e-10,
                stagnation_rel_tolerance=1e-10,
            ),
            "epsilon должен быть > 0",
        ),
        (
            SolverConfig(
                epsilon=1e-3,
                max_iterations=10,
                line_search_min_lambda=0.5,
                line_search_max_lambda=0.5,
                line_search_tolerance=1e-5,
                line_search_max_iterations=120,
                direction_zero_tolerance=1e-12,
                stagnation_abs_tolerance=1e-10,
                stagnation_rel_tolerance=1e-10,
            ),
            "line_search_min_lambda должен быть < line_search_max_lambda",
        ),
    ),
)
def test_rosenbrock_minimize_rejects_invalid_solver_config(
    config: SolverConfig,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        rosenbrock_minimize(
            objective=lambda vector: sum(value * value for value in vector),
            start_point=(1.0, 1.0),
            config=config,
        )
