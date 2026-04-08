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
    """Линейный поиск шага вдоль направления через золотое сечение."""
    current_value = objective(point)
    search_upper_bound: float | None = None

    def line_objective(step: float) -> float:
        candidate = _add(point, direction, step)
        return objective(candidate)

    step = max(initial_step, min_step)
    last_improving_step = 0.0
    last_improving_value = current_value
    expanded = False

    for _ in range(max(expand_limit, 1)):
        try:
            candidate_value = line_objective(step)
        except OverflowError:
            break
        if (maximize and candidate_value > last_improving_value) or (not maximize and candidate_value < last_improving_value):
            last_improving_step = step
            last_improving_value = candidate_value
            step *= 2.0
            expanded = True
            continue
        search_upper_bound = step
        break

    if search_upper_bound is None and expanded:
        search_upper_bound = last_improving_step

    if search_upper_bound is None:
        step = max(initial_step, min_step)
        while step >= min_step:
            try:
                candidate_value = line_objective(step)
            except OverflowError:
                break
            if (maximize and candidate_value > current_value) or (not maximize and candidate_value < current_value):
                search_upper_bound = max(initial_step, step)
                break
            step /= 2.0

    if search_upper_bound is None or search_upper_bound <= 0.0:
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
    elif best_step > initial_step:
        decision = "accepted_after_expansion"
    else:
        decision = "accepted_after_reduction"
    return best_point, best_value, best_step, True, decision


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
    """Метод сопряжённых градиентов по схеме Флетчера-Ривса для max/min цели."""
    started = time.perf_counter()
    multiplier = _goal_multiplier(config.goal)
    maximize = config.goal == "max"
    records: list[IterationRecord] = []
    dimension = len(start_point)
    x_k = start_point
    current_point = start_point
    total_iterations = 0
    cycle_index = 1

    while total_iterations < config.max_iterations:
        y_j = x_k
        gradient = finite_difference_gradient(objective, y_j, config.gradient_step)
        signed_gradient = (multiplier * gradient[0], multiplier * gradient[1])
        direction = signed_gradient
        direction_index = 1

        while direction_index <= dimension and total_iterations < config.max_iterations:
            if (time.perf_counter() - started) > config.timeout_seconds:
                return OptimizationResult(
                    method_name="conjugate_gradient_ascent",
                    start_point=start_point,
                    optimum_point=y_j,
                    optimum_value=objective(y_j),
                    iterations_count=total_iterations,
                    records=tuple(records),
                    success=False,
                    stop_reason="timeout",
                )

            value = objective(y_j)
            grad_norm = _vector_norm(gradient)

            if grad_norm <= config.epsilon:
                records.append(
                    IterationRecord(
                        k=total_iterations,
                        cycle_index=cycle_index,
                        direction_index=direction_index,
                        cycle_start_point=x_k,
                        point=y_j,
                        value=value,
                        gradient=gradient,
                        step_size=0.0,
                        direction=direction,
                        restart_direction=False,
                    )
                )
                return OptimizationResult(
                    method_name="conjugate_gradient_ascent",
                    start_point=start_point,
                    optimum_point=y_j,
                    optimum_value=value,
                    iterations_count=total_iterations + 1,
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
                expand_limit=config.max_step_expansions,
                maximize=maximize,
            )

            if not improved:
                records.append(
                    IterationRecord(
                        k=total_iterations,
                        cycle_index=cycle_index,
                        direction_index=direction_index,
                        cycle_start_point=x_k,
                        point=y_j,
                        value=value,
                        gradient=gradient,
                        step_size=0.0,
                        direction=direction,
                        next_point=None,
                        next_value=None,
                        beta=None,
                        restart_direction=False,
                    )
                )
                return OptimizationResult(
                    method_name="conjugate_gradient_ascent",
                    start_point=start_point,
                    optimum_point=y_j,
                    optimum_value=value,
                    iterations_count=total_iterations + 1,
                    records=tuple(records),
                    success=False,
                    stop_reason="no_improving_step",
                )

            restart = direction_index == dimension
            beta: float | None = None
            new_gradient = finite_difference_gradient(objective, new_point, config.gradient_step)
            next_direction: Point2D | None = None

            if not restart:
                new_signed_gradient = (multiplier * new_gradient[0], multiplier * new_gradient[1])
                denominator = max(_dot(signed_gradient, signed_gradient), 1e-14)
                beta = _dot(new_signed_gradient, new_signed_gradient) / denominator
                next_direction = _add(new_signed_gradient, direction, beta)

            records.append(
                IterationRecord(
                    k=total_iterations,
                    cycle_index=cycle_index,
                    direction_index=direction_index,
                    cycle_start_point=x_k,
                    point=y_j,
                    value=value,
                    gradient=gradient,
                    step_size=step_size,
                    direction=direction,
                    beta=beta,
                    next_point=new_point,
                    next_value=new_value,
                    restart_direction=restart,
                )
            )

            total_iterations += 1
            current_point = new_point
            if restart:
                x_k = new_point
                cycle_index += 1
                break

            y_j = new_point
            gradient = new_gradient
            signed_gradient = (multiplier * gradient[0], multiplier * gradient[1])
            direction = next_direction if next_direction is not None else signed_gradient
            direction_index += 1

        else:
            continue

        continue

    return OptimizationResult(
        method_name="conjugate_gradient_ascent",
        start_point=start_point,
        optimum_point=current_point,
        optimum_value=objective(current_point),
        iterations_count=config.max_iterations,
        records=tuple(records),
        success=False,
        stop_reason="max_iterations_reached",
    )
