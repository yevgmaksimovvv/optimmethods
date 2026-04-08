"""Численные методы ЛР3: градиентный и сопряжённых градиентов."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from lr3.domain.models import Goal, GradientStepDecision, IterationRecord, MethodConfig, OptimizationResult, Point2D

ObjectiveFn = Callable[[Point2D], float]


def finite_difference_gradient(
    objective: ObjectiveFn,
    point: Point2D,
    step: float,
) -> Point2D:
    """Центральная разностная аппроксимация градиента."""
    x1, x2 = point
    step_x1 = max(step, 1e-8 * max(1.0, abs(x1)))
    step_x2 = max(step, 1e-8 * max(1.0, abs(x2)))
    dx = (objective((x1 + step_x1, x2)) - objective((x1 - step_x1, x2))) / (2.0 * step_x1)
    dy = (objective((x1, x2 + step_x2)) - objective((x1, x2 - step_x2))) / (2.0 * step_x2)
    return (dx, dy)


def _vector_norm(point: Point2D) -> float:
    return math.hypot(point[0], point[1])


def _add(a: Point2D, b: Point2D, scale: float = 1.0) -> Point2D:
    return (a[0] + scale * b[0], a[1] + scale * b[1])


def _dot(a: Point2D, b: Point2D) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _goal_multiplier(goal: Goal) -> float:
    return 1.0 if goal == "max" else -1.0


def _line_search(
    objective: ObjectiveFn,
    point: Point2D,
    direction: Point2D,
    initial_step: float,
    min_step: float,
    expand_limit: int,
    maximize: bool,
) -> tuple[Point2D, float, float, bool, GradientStepDecision | None]:
    """Линейный поиск шага вдоль направления для max/min цели."""
    current_value = objective(point)
    best_point = point
    best_value = current_value
    best_step = 0.0

    step = max(initial_step, min_step)
    for _ in range(max(expand_limit, 1)):
        candidate = _add(point, direction, step)
        try:
            candidate_value = objective(candidate)
        except OverflowError:
            break
        if (maximize and candidate_value > best_value) or (not maximize and candidate_value < best_value):
            best_point = candidate
            best_value = candidate_value
            best_step = step
            step *= 2.0
        else:
            break

    if best_step > 0.0:
        if math.isclose(best_step, initial_step, rel_tol=1e-9, abs_tol=1e-12):
            return best_point, best_value, best_step, True, "accepted_as_is"
        return best_point, best_value, best_step, True, "accepted_after_expansion"

    step = max(initial_step, min_step)
    while step >= min_step:
        candidate = _add(point, direction, step)
        try:
            candidate_value = objective(candidate)
        except OverflowError:
            break
        if (maximize and candidate_value > current_value) or (not maximize and candidate_value < current_value):
            if math.isclose(step, initial_step, rel_tol=1e-9, abs_tol=1e-12):
                return candidate, candidate_value, step, True, "accepted_as_is"
            return candidate, candidate_value, step, True, "accepted_after_reduction"
        step /= 2.0

    return point, current_value, 0.0, False, None


def gradient_ascent(
    objective: ObjectiveFn,
    start_point: Point2D,
    config: MethodConfig,
) -> OptimizationResult:
    """Градиентный метод с адаптивным выбором шага для max/min цели."""
    started = time.perf_counter()
    point = start_point
    records: list[IterationRecord] = []
    multiplier = _goal_multiplier(config.goal)
    maximize = config.goal == "max"

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
        records.append(
            IterationRecord(
                k=k,
                point=point,
                value=value,
                gradient=gradient,
                step_size=0.0,
                gradient_step_decision="precision_reached" if grad_norm <= config.epsilon else None,
            )
        )

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

        direction = (multiplier * gradient[0], multiplier * gradient[1])
        new_point, new_value, step_size, improved, step_decision = _line_search(
            objective=objective,
            point=point,
            direction=direction,
            initial_step=config.initial_step,
            min_step=config.min_step,
            expand_limit=config.max_step_expansions,
            maximize=maximize,
        )

        records[-1] = IterationRecord(
            k=k,
            point=point,
            value=value,
            gradient=gradient,
            step_size=step_size,
            gradient_step_decision=step_decision if improved else "no_improving_step",
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
    """Нелинейный метод сопряжённых градиентов (Fletcher-Reeves) для max/min цели."""
    started = time.perf_counter()
    point = start_point
    multiplier = _goal_multiplier(config.goal)
    maximize = config.goal == "max"
    gradient = finite_difference_gradient(objective, point, config.gradient_step)
    signed_gradient = (multiplier * gradient[0], multiplier * gradient[1])
    direction = signed_gradient
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

        current_point = point
        current_gradient = gradient
        current_signed_gradient = signed_gradient
        current_direction = direction
        value = objective(current_point)
        grad_norm = _vector_norm(current_gradient)

        if grad_norm <= config.epsilon:
            records.append(
                IterationRecord(
                    k=k,
                    point=current_point,
                    value=value,
                    gradient=current_gradient,
                    step_size=0.0,
                    direction=current_direction,
                )
            )
            return OptimizationResult(
                method_name="conjugate_gradient_ascent",
                start_point=start_point,
                optimum_point=current_point,
                optimum_value=value,
                iterations_count=k + 1,
                records=tuple(records),
                success=True,
                stop_reason="gradient_norm_reached",
            )

        new_point, new_value, step_size, improved, _ = _line_search(
            objective=objective,
            point=current_point,
            direction=current_direction,
            initial_step=config.initial_step,
            min_step=config.min_step,
            expand_limit=config.max_step_expansions,
            maximize=maximize,
        )

        if not improved:
            records.append(
                IterationRecord(
                    k=k,
                    point=current_point,
                    value=value,
                    gradient=current_gradient,
                    step_size=0.0,
                    direction=current_direction,
                    next_point=None,
                    next_value=None,
                    beta=None,
                    restart_direction=False,
                )
            )
            return OptimizationResult(
                method_name="conjugate_gradient_ascent",
                start_point=start_point,
                optimum_point=current_point,
                optimum_value=value,
                iterations_count=k + 1,
                records=tuple(records),
                success=False,
                stop_reason="no_improving_step",
            )

        new_gradient = finite_difference_gradient(objective, new_point, config.gradient_step)
        signed_gradient = (multiplier * new_gradient[0], multiplier * new_gradient[1])
        denominator = max(_dot(current_signed_gradient, current_signed_gradient), 1e-14)
        beta = _dot(signed_gradient, signed_gradient) / denominator
        updated_direction = _add(signed_gradient, current_direction, beta)
        restart = False

        if _dot(updated_direction, signed_gradient) <= 0:
            updated_direction = signed_gradient
            restart = True

        records.append(
            IterationRecord(
                k=k,
                point=current_point,
                value=value,
                gradient=current_gradient,
                step_size=step_size,
                direction=current_direction,
                beta=beta,
                next_point=new_point,
                next_value=new_value,
                restart_direction=restart,
            )
        )

        point = new_point
        gradient = new_gradient
        signed_gradient = (multiplier * gradient[0], multiplier * gradient[1])
        direction = updated_direction
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
