"""Численные методы ЛР3: градиентный и сопряжённых градиентов."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from lr1.domain.search import golden_section_search
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


def _goal_multiplier(goal: Goal) -> float:
    return 1.0 if goal == "max" else -1.0


def _line_search(
    objective: ObjectiveFn,
    point: Point2D,
    direction: Point2D,
    initial_step: float,
    min_step: float,
    maximize: bool,
) -> tuple[Point2D, float, float, bool, GradientStepDecision | None]:
    """Линейный поиск шага вдоль направления на фиксированном отрезке через золотое сечение."""
    current_value = objective(point)

    def line_objective(step: float) -> float:
        candidate = _add(point, direction, step)
        return objective(candidate)

    search_upper_bound = max(initial_step, min_step)
    if search_upper_bound <= 0.0:
        return point, current_value, 0.0, False, None

    search_result = golden_section_search(
        func=line_objective,
        a=0.0,
        b=search_upper_bound,
        eps=min_step,
        l=min_step,
        kind="max" if maximize else "min",
    )
    best_step = search_result.x_opt
    best_value = search_result.f_opt

    if not ((maximize and best_value > current_value) or (not maximize and best_value < current_value)):
        return point, current_value, 0.0, False, None

    best_point = _add(point, direction, best_step)
    if math.isclose(best_step, initial_step, rel_tol=1e-6, abs_tol=min_step):
        decision: GradientStepDecision = "accepted_as_is"
    else:
        decision = "accepted_after_reduction"
    return best_point, best_value, best_step, True, decision


def gradient_ascent(
    objective: ObjectiveFn,
    start_point: Point2D,
    config: MethodConfig,
) -> OptimizationResult:
    """Градиентный метод с фиксированным шагом и делением шага пополам при неудаче."""
    started = time.perf_counter()
    point = start_point
    records: list[IterationRecord] = []
    multiplier = _goal_multiplier(config.goal)
    maximize = config.goal == "max"
    step_size = max(config.initial_step, config.min_step)

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
        precision_reached = all(abs(component) <= config.epsilon for component in gradient)
        records.append(
            IterationRecord(
                k=k,
                point=point,
                value=value,
                gradient=gradient,
                step_size=0.0,
                gradient_step_decision="precision_reached" if precision_reached else None,
            )
        )

        if precision_reached:
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
        accepted_step = step_size
        new_point = point
        new_value = value
        improved = False
        step_decision: GradientStepDecision | None = None

        while accepted_step >= config.min_step:
            candidate_point = _add(point, direction, accepted_step)
            try:
                candidate_value = objective(candidate_point)
            except OverflowError:
                candidate_value = math.inf if not maximize else -math.inf

            improved = (
                candidate_value > value if maximize else candidate_value < value
            )
            if improved:
                new_point = candidate_point
                new_value = candidate_value
                step_decision = (
                    "accepted_as_is"
                    if math.isclose(accepted_step, step_size, rel_tol=1e-12, abs_tol=config.min_step)
                    else "accepted_after_reduction"
                )
                break
            accepted_step /= 2.0

        records[-1] = IterationRecord(
            k=k,
            point=point,
            value=value,
            gradient=gradient,
            step_size=accepted_step if improved else 0.0,
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

        step_size = accepted_step
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
    """Метод сопряжённых градиентов с рестартом через размерность пространства."""
    started = time.perf_counter()
    multiplier = _goal_multiplier(config.goal)
    maximize = config.goal == "max"
    records: list[IterationRecord] = []
    dimension = len(start_point)
    x_k = start_point
    cycle_index = 1

    while len(records) < config.max_iterations:
        if (time.perf_counter() - started) > config.timeout_seconds:
            return OptimizationResult(
                method_name="conjugate_gradient_ascent",
                start_point=start_point,
                optimum_point=x_k,
                optimum_value=objective(x_k),
                iterations_count=len(records),
                records=tuple(records),
                success=False,
                stop_reason="timeout",
            )

        y_j = x_k
        gradient = finite_difference_gradient(objective, y_j, config.gradient_step)
        direction = (multiplier * gradient[0], multiplier * gradient[1])
        direction_beta: float | None = None

        for direction_index in range(1, dimension + 1):
            if len(records) >= config.max_iterations:
                return OptimizationResult(
                    method_name="conjugate_gradient_ascent",
                    start_point=start_point,
                    optimum_point=y_j,
                    optimum_value=objective(y_j),
                    iterations_count=len(records),
                    records=tuple(records),
                    success=False,
                    stop_reason="max_iterations_reached",
                )

            if (time.perf_counter() - started) > config.timeout_seconds:
                return OptimizationResult(
                    method_name="conjugate_gradient_ascent",
                    start_point=start_point,
                    optimum_point=y_j,
                    optimum_value=objective(y_j),
                    iterations_count=len(records),
                    records=tuple(records),
                    success=False,
                    stop_reason="timeout",
                )

            value = objective(y_j)
            gradient_norm = _vector_norm(gradient)
            if gradient_norm <= config.epsilon:
                records.append(
                    IterationRecord(
                        k=len(records),
                        cycle_index=cycle_index,
                        direction_index=direction_index,
                        cycle_start_point=x_k,
                        point=y_j,
                        value=value,
                        gradient=gradient,
                        step_size=0.0,
                        direction=None,
                        beta=direction_beta,
                        restart_direction=False,
                    )
                )
                return OptimizationResult(
                    method_name="conjugate_gradient_ascent",
                    start_point=start_point,
                    optimum_point=y_j,
                    optimum_value=value,
                    iterations_count=len(records),
                    records=tuple(records),
                    success=True,
                    stop_reason="gradient_norm_reached",
                )

            new_point, new_value, step_size, improved, _ = _line_search(
                objective=objective,
                point=y_j,
                direction=direction,
                initial_step=config.initial_step,
                min_step=config.min_step,
                maximize=maximize,
            )
            records.append(
                IterationRecord(
                    k=len(records),
                    cycle_index=cycle_index,
                    direction_index=direction_index,
                    cycle_start_point=x_k,
                    point=y_j,
                    value=value,
                    gradient=gradient,
                    step_size=step_size,
                    direction=direction,
                    beta=direction_beta,
                    next_point=new_point,
                    next_value=new_value,
                    restart_direction=direction_index == dimension,
                )
            )

            if not improved or step_size <= 0.0:
                return OptimizationResult(
                    method_name="conjugate_gradient_ascent",
                    start_point=start_point,
                    optimum_point=y_j,
                    optimum_value=value,
                    iterations_count=len(records),
                    records=tuple(records),
                    success=False,
                    stop_reason="no_improving_step",
                )

            if direction_index == dimension:
                y_j = new_point
                break

            next_gradient = finite_difference_gradient(objective, new_point, config.gradient_step)
            prev_gradient_norm_sq = gradient[0] * gradient[0] + gradient[1] * gradient[1]
            next_gradient_norm_sq = next_gradient[0] * next_gradient[0] + next_gradient[1] * next_gradient[1]
            beta = 0.0 if prev_gradient_norm_sq <= 0.0 else next_gradient_norm_sq / prev_gradient_norm_sq
            direction = (
                multiplier * next_gradient[0] + beta * direction[0],
                multiplier * next_gradient[1] + beta * direction[1],
            )
            y_j = new_point
            gradient = next_gradient
            direction_beta = beta

        x_k = y_j
        cycle_index += 1

    return OptimizationResult(
        method_name="conjugate_gradient_ascent",
        start_point=start_point,
        optimum_point=x_k,
        optimum_value=objective(x_k),
        iterations_count=config.max_iterations,
        records=tuple(records),
        success=False,
        stop_reason="max_iterations_reached",
    )
