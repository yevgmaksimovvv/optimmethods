"""Численные методы ЛР3: градиентный и сопряжённых градиентов."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from lr3.domain.models import IterationRecord, MethodConfig, OptimizationResult, Point2D

ObjectiveFn = Callable[[Point2D], float]


def finite_difference_gradient(
    objective: ObjectiveFn,
    point: Point2D,
    step: float,
) -> Point2D:
    """Центральная разностная аппроксимация градиента."""
    x1, x2 = point
    dx = (objective((x1 + step, x2)) - objective((x1 - step, x2))) / (2.0 * step)
    dy = (objective((x1, x2 + step)) - objective((x1, x2 - step))) / (2.0 * step)
    return (dx, dy)


def _vector_norm(point: Point2D) -> float:
    return math.hypot(point[0], point[1])


def _add(a: Point2D, b: Point2D, scale: float = 1.0) -> Point2D:
    return (a[0] + scale * b[0], a[1] + scale * b[1])


def _dot(a: Point2D, b: Point2D) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _maximize_line_search(
    objective: ObjectiveFn,
    point: Point2D,
    direction: Point2D,
    initial_step: float,
    min_step: float,
    expand_limit: int,
) -> tuple[Point2D, float, float, bool]:
    """Линейный поиск шага для максимизации вдоль направления."""
    current_value = objective(point)
    best_point = point
    best_value = current_value
    best_step = 0.0

    step = max(initial_step, min_step)
    for _ in range(max(expand_limit, 1)):
        candidate = _add(point, direction, step)
        candidate_value = objective(candidate)
        if candidate_value > best_value:
            best_point = candidate
            best_value = candidate_value
            best_step = step
            step *= 2.0
        else:
            break

    if best_step > 0.0:
        return best_point, best_value, best_step, True

    step = max(initial_step, min_step)
    while step >= min_step:
        candidate = _add(point, direction, step)
        candidate_value = objective(candidate)
        if candidate_value > current_value:
            return candidate, candidate_value, step, True
        step /= 2.0

    return point, current_value, 0.0, False


def gradient_ascent(
    objective: ObjectiveFn,
    start_point: Point2D,
    config: MethodConfig,
) -> OptimizationResult:
    """Градиентный подъем с адаптивным выбором шага."""
    started = time.perf_counter()
    point = start_point
    records: list[IterationRecord] = []

    for k in range(config.max_iterations):
        if (time.perf_counter() - started) > config.timeout_seconds:
            return OptimizationResult(
                method_name="gradient_ascent",
                start_point=start_point,
                optimum_point=point,
                optimum_value=objective(point),
                iterations_count=k,
                records=tuple(records),
                success=False,
                stop_reason="timeout",
            )

        gradient = finite_difference_gradient(objective, point, config.gradient_step)
        value = objective(point)
        grad_norm = _vector_norm(gradient)
        records.append(IterationRecord(k=k, point=point, value=value, gradient=gradient, step_size=0.0))

        if grad_norm <= config.epsilon:
            return OptimizationResult(
                method_name="gradient_ascent",
                start_point=start_point,
                optimum_point=point,
                optimum_value=value,
                iterations_count=k + 1,
                records=tuple(records),
                success=True,
                stop_reason="gradient_norm_reached",
            )

        new_point, new_value, step_size, improved = _maximize_line_search(
            objective=objective,
            point=point,
            direction=gradient,
            initial_step=config.initial_step,
            min_step=config.min_step,
            expand_limit=config.max_step_expansions,
        )

        records[-1] = IterationRecord(
            k=k,
            point=point,
            value=value,
            gradient=gradient,
            step_size=step_size,
        )

        if not improved:
            return OptimizationResult(
                method_name="gradient_ascent",
                start_point=start_point,
                optimum_point=point,
                optimum_value=value,
                iterations_count=k + 1,
                records=tuple(records),
                success=False,
                stop_reason="no_improving_step",
            )

        point = new_point
        _ = new_value

    return OptimizationResult(
        method_name="gradient_ascent",
        start_point=start_point,
        optimum_point=point,
        optimum_value=objective(point),
        iterations_count=config.max_iterations,
        records=tuple(records),
        success=False,
        stop_reason="max_iterations_reached",
    )


def conjugate_gradient_ascent(
    objective: ObjectiveFn,
    start_point: Point2D,
    config: MethodConfig,
) -> OptimizationResult:
    """Нелинейный метод сопряжённых градиентов (Fletcher-Reeves) для максимизации."""
    started = time.perf_counter()
    point = start_point
    gradient = finite_difference_gradient(objective, point, config.gradient_step)
    direction = gradient
    records: list[IterationRecord] = []

    for k in range(config.max_iterations):
        if (time.perf_counter() - started) > config.timeout_seconds:
            return OptimizationResult(
                method_name="conjugate_gradient_ascent",
                start_point=start_point,
                optimum_point=point,
                optimum_value=objective(point),
                iterations_count=k,
                records=tuple(records),
                success=False,
                stop_reason="timeout",
            )

        value = objective(point)
        grad_norm = _vector_norm(gradient)
        records.append(IterationRecord(k=k, point=point, value=value, gradient=gradient, step_size=0.0))

        if grad_norm <= config.epsilon:
            return OptimizationResult(
                method_name="conjugate_gradient_ascent",
                start_point=start_point,
                optimum_point=point,
                optimum_value=value,
                iterations_count=k + 1,
                records=tuple(records),
                success=True,
                stop_reason="gradient_norm_reached",
            )

        new_point, new_value, step_size, improved = _maximize_line_search(
            objective=objective,
            point=point,
            direction=direction,
            initial_step=config.initial_step,
            min_step=config.min_step,
            expand_limit=config.max_step_expansions,
        )

        records[-1] = IterationRecord(
            k=k,
            point=point,
            value=value,
            gradient=gradient,
            step_size=step_size,
        )

        if not improved:
            return OptimizationResult(
                method_name="conjugate_gradient_ascent",
                start_point=start_point,
                optimum_point=point,
                optimum_value=value,
                iterations_count=k + 1,
                records=tuple(records),
                success=False,
                stop_reason="no_improving_step",
            )

        new_gradient = finite_difference_gradient(objective, new_point, config.gradient_step)
        denominator = max(_dot(gradient, gradient), 1e-14)
        beta = _dot(new_gradient, new_gradient) / denominator
        direction = _add(new_gradient, direction, beta)

        if _dot(direction, new_gradient) <= 0:
            direction = new_gradient

        point = new_point
        gradient = new_gradient
        _ = new_value

    return OptimizationResult(
        method_name="conjugate_gradient_ascent",
        start_point=start_point,
        optimum_point=point,
        optimum_value=objective(point),
        iterations_count=config.max_iterations,
        records=tuple(records),
        success=False,
        stop_reason="max_iterations_reached",
    )
