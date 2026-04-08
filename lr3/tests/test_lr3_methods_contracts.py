"""Дополнительные контрактные тесты численных методов ЛР3."""

from __future__ import annotations

import math
import time

import pytest

from lr3.domain.methods import conjugate_gradient_ascent, gradient_ascent
from lr3.domain.models import MethodConfig, Point2D


def _config(
    *,
    epsilon: float = 1e-6,
    max_iterations: int = 250,
    initial_step: float = 0.1,
    timeout_seconds: float = 2.0,
    goal: str = "max",
    min_step: float = 1e-8,
    gradient_step: float = 1e-6,
    max_step_expansions: int = 16,
) -> MethodConfig:
    return MethodConfig(
        epsilon=epsilon,
        max_iterations=max_iterations,
        initial_step=initial_step,
        min_step=min_step,
        timeout_seconds=timeout_seconds,
        goal=goal,
        gradient_step=gradient_step,
        max_step_expansions=max_step_expansions,
    )


def test_gradient_ascent_converges_to_quadratic_peak() -> None:
    def objective(point: Point2D) -> float:
        x1, x2 = point
        return -(x1 - 1.0) ** 2 - 2.0 * (x2 + 2.0) ** 2

    start_point: Point2D = (7.0, -9.0)

    result = gradient_ascent(objective, start_point, _config())

    assert result.success
    assert result.stop_reason == "gradient_norm_reached"
    assert result.iterations_count >= 1
    assert len(result.records) == result.iterations_count
    assert result.records[0].point == pytest.approx(start_point)
    assert any(record.step_size > 0.0 for record in result.records)
    assert math.isfinite(result.optimum_point[0])
    assert math.isfinite(result.optimum_point[1])
    assert result.optimum_point == pytest.approx((1.0, -2.0), abs=5e-1)
    assert result.optimum_value == pytest.approx(0.0, abs=1e-2)


def test_conjugate_gradient_ascent_converges_to_peak() -> None:
    def objective(point: Point2D) -> float:
        x1, x2 = point
        return -(x1 - 3.0) ** 2 - 2.0 * (x2 - 4.0) ** 2

    start_point: Point2D = (0.0, 0.0)

    result = conjugate_gradient_ascent(objective, start_point, _config(initial_step=0.2))

    assert result.success
    assert result.stop_reason == "gradient_norm_reached"
    assert result.iterations_count >= 1
    assert len(result.records) == result.iterations_count
    assert result.records[0].point == pytest.approx(start_point)
    assert any(record.step_size > 0.0 for record in result.records)
    assert math.isfinite(result.optimum_value)
    assert result.optimum_point == pytest.approx((3.0, 4.0), abs=5e-1)
    assert result.optimum_value == pytest.approx(0.0, abs=1e-2)


def test_gradient_ascent_supports_minimum_goal() -> None:
    def objective(point: Point2D) -> float:
        x1, x2 = point
        return (x1 - 1.0) ** 2 + 2.0 * (x2 + 2.0) ** 2

    start_point: Point2D = (7.0, -9.0)

    result = gradient_ascent(objective, start_point, _config(goal="min"))

    assert result.success
    assert result.stop_reason == "gradient_norm_reached"
    assert result.optimum_point == pytest.approx((1.0, -2.0), abs=5e-1)
    assert result.optimum_value == pytest.approx(0.0, abs=1e-2)


def test_conjugate_gradient_ascent_supports_minimum_goal() -> None:
    def objective(point: Point2D) -> float:
        x1, x2 = point
        return (x1 - 3.0) ** 2 + 2.0 * (x2 - 4.0) ** 2

    result = conjugate_gradient_ascent(objective, (0.0, 0.0), _config(initial_step=0.2, goal="min"))

    assert result.success
    assert result.stop_reason == "gradient_norm_reached"
    assert result.optimum_point == pytest.approx((3.0, 4.0), abs=5e-1)
    assert result.optimum_value == pytest.approx(0.0, abs=1e-2)


def test_conjugate_gradient_history_carries_step_metadata() -> None:
    def objective(point: Point2D) -> float:
        x1, x2 = point
        return -(x1 - 3.0) ** 2 - 2.0 * (x2 - 4.0) ** 2

    result = conjugate_gradient_ascent(objective, (0.0, 0.0), _config(initial_step=0.2))

    record = result.records[0]

    assert record.direction is not None
    assert record.beta is not None
    assert record.next_point is not None
    assert record.next_value is not None
    assert record.cycle_index == 1
    assert record.direction_index == 1
    assert record.cycle_start_point == pytest.approx((0.0, 0.0))
    assert record.restart_direction is False
    assert record.direction == pytest.approx(record.gradient)
    assert record.next_point == pytest.approx(
        (
            record.point[0] + record.step_size * record.direction[0],
            record.point[1] + record.step_size * record.direction[1],
        )
    )
    assert record.next_value == pytest.approx(objective(record.next_point))


def test_conjugate_gradient_restarts_after_dimension_steps() -> None:
    def objective(point: Point2D) -> float:
        x1, x2 = point
        return -(x1 - 3.0) ** 2 - 2.0 * (x2 - 4.0) ** 2

    result = conjugate_gradient_ascent(objective, (0.0, 0.0), _config(initial_step=0.2))

    assert len(result.records) >= 2
    first_record = result.records[0]
    second_record = result.records[1]

    assert first_record.cycle_index == 1
    assert first_record.direction_index == 1
    assert first_record.restart_direction is False
    assert second_record.cycle_index == 1
    assert second_record.direction_index == 2
    assert second_record.restart_direction is True


def test_gradient_ascent_stops_at_iteration_limit() -> None:
    def objective(point: Point2D) -> float:
        x1, x2 = point
        return -(x1 - 1.0) ** 2 - 2.0 * (x2 + 2.0) ** 2

    result = gradient_ascent(
        objective,
        start_point=(7.0, -9.0),
        config=_config(epsilon=1e-12, max_iterations=1),
    )

    assert not result.success
    assert result.stop_reason == "max_iterations_reached"
    assert result.iterations_count == 1
    assert len(result.records) == 1
    assert math.isfinite(result.optimum_value)
    assert result.records[0].step_size > 0.0


def test_gradient_ascent_times_out_on_slow_objective() -> None:
    def objective(point: Point2D) -> float:
        time.sleep(0.005)
        x1, x2 = point
        return -(x1 - 1.0) ** 2 - (x2 + 2.0) ** 2

    result = gradient_ascent(
        objective,
        start_point=(7.0, -9.0),
        config=_config(epsilon=1e-12, max_iterations=250, timeout_seconds=0.001),
    )

    assert not result.success
    assert result.stop_reason == "timeout"
    assert result.iterations_count == 1
    assert len(result.records) == 1
    assert math.isfinite(result.optimum_value)


def test_gradient_ascent_does_not_fake_success_on_unbounded_maximum() -> None:
    def objective(point: Point2D) -> float:
        x1, x2 = point
        return x1**2 + x2**2 - x1 * x2 + x1 - 2.0 * x2

    result = gradient_ascent(
        objective,
        start_point=(0.0, 0.0),
        config=_config(goal="max", max_iterations=40, timeout_seconds=1.0),
    )

    assert not result.success
    assert result.stop_reason != "gradient_norm_reached"
    assert result.iterations_count > 0
