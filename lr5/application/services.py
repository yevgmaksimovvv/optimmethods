"""Прикладной слой ЛР5: постановка задачи и запуск барьерного метода."""

from __future__ import annotations

import logging
import math
from typing import Final

from optim_core.parsing import parse_localized_float_sequence

from lr2.application.services import DEFAULT_SOLVER_CONFIG
from lr2.domain.models import SolverConfig, Vector
from lr2.domain.rosenbrock import rosenbrock_minimize
from lr5.domain.barrier import (
    LOG_BARRIER,
    RECIPROCAL_BARRIER,
    barrier_metric,
    barrier_value,
    is_strictly_feasible,
)
from lr5.domain.models import (
    BARRIER_STATUS_ERROR,
    BARRIER_STATUS_SUCCESS,
    BARRIER_STATUS_WARNING,
    BarrierConstraint,
    BarrierIterationResult,
    BarrierMethodConfig,
    BarrierProblem,
    BarrierResult,
)

logger = logging.getLogger("lr5.service")

POLYNOMIAL_DIMENSION: Final[int] = 2
DEFAULT_MU0 = 10.0
DEFAULT_BETA = 0.1
DEFAULT_EPSILON_OUTER = 1e-3
DEFAULT_MAX_OUTER_ITERATIONS = 20
DEFAULT_BARRIER_KIND: str = RECIPROCAL_BARRIER
DEFAULT_FEASIBILITY_TOLERANCE = 1e-12
DEFAULT_INNER_EPSILON = 1e-4


def variant_2_problem() -> BarrierProblem:
    """Возвращает постановку задачи варианта 2."""

    def objective(vector: Vector) -> float:
        x1, x2 = vector
        return (x1 - 5.0) ** 2 + (x2 - 3.0) ** 2

    constraints = (
        BarrierConstraint(title="g1(x) = -x1 + 2x2 - 4", evaluator=lambda vector: -vector[0] + 2.0 * vector[1] - 4.0),
        BarrierConstraint(title="g2(x) = x1 + x2 - 3", evaluator=lambda vector: vector[0] + vector[1] - 3.0),
    )
    return BarrierProblem(
        title="Вариант 2",
        objective_title="F(x1, x2) = (x1 - 5)^2 + (x2 - 3)^2",
        objective=objective,
        constraints=constraints,
    )


def build_inner_solver_config(*, epsilon: float = DEFAULT_INNER_EPSILON, max_iterations: int | None = None) -> SolverConfig:
    payload = dict(DEFAULT_SOLVER_CONFIG)
    if max_iterations is not None:
        payload["max_iterations"] = max_iterations
    return SolverConfig(
        epsilon=epsilon,
        max_iterations=int(payload["max_iterations"]),
        line_search_min_lambda=float(payload["line_search_min_lambda"]),
        line_search_max_lambda=float(payload["line_search_max_lambda"]),
        line_search_tolerance=float(payload["line_search_tolerance"]),
        line_search_max_iterations=int(payload["line_search_max_iterations"]),
        direction_zero_tolerance=float(payload["direction_zero_tolerance"]),
        stagnation_abs_tolerance=float(payload["stagnation_abs_tolerance"]),
        stagnation_rel_tolerance=float(payload["stagnation_rel_tolerance"]),
    )


def parse_vector(raw: str) -> tuple[float, float]:
    values = parse_localized_float_sequence(raw, "x", separator=";")
    if len(values) != POLYNOMIAL_DIMENSION:
        raise ValueError(f"Ожидался вектор размерности {POLYNOMIAL_DIMENSION}, получено {len(values)}")
    return float(values[0]), float(values[1])


def build_method_config(
    *,
    mu0: float = DEFAULT_MU0,
    beta: float = DEFAULT_BETA,
    epsilon_outer: float = DEFAULT_EPSILON_OUTER,
    max_outer_iterations: int = DEFAULT_MAX_OUTER_ITERATIONS,
    barrier_kind: str = DEFAULT_BARRIER_KIND,
    inner_epsilon: float = DEFAULT_INNER_EPSILON,
    inner_max_iterations: int | None = None,
    feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
) -> BarrierMethodConfig:
    if mu0 <= 0.0:
        raise ValueError("mu0 должен быть > 0.")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta должен быть в диапазоне (0, 1).")
    if epsilon_outer <= 0.0:
        raise ValueError("epsilon внешнего цикла должен быть > 0.")
    if max_outer_iterations < 1:
        raise ValueError("max_outer_iterations должен быть >= 1.")
    return BarrierMethodConfig(
        mu0=mu0,
        beta=beta,
        epsilon_outer=epsilon_outer,
        max_outer_iterations=max_outer_iterations,
        barrier_kind=barrier_kind,
        inner_solver_config=build_inner_solver_config(
            epsilon=inner_epsilon,
            max_iterations=inner_max_iterations,
        ),
        feasibility_tolerance=feasibility_tolerance,
    )


def run_barrier_method(
    problem: BarrierProblem,
    start_point: Vector,
    config: BarrierMethodConfig,
) -> BarrierResult:
    if len(start_point) != POLYNOMIAL_DIMENSION:
        raise ValueError(f"Ожидалась точка размерности {POLYNOMIAL_DIMENSION}, получено {len(start_point)}")
    if config.mu0 <= 0.0:
        raise ValueError("mu0 должен быть > 0.")
    if not (0.0 < config.beta < 1.0):
        raise ValueError("beta должен быть в диапазоне (0, 1).")
    if config.epsilon_outer <= 0.0:
        raise ValueError("epsilon внешнего цикла должен быть > 0.")
    if config.max_outer_iterations < 1:
        raise ValueError("max_outer_iterations должен быть >= 1.")
    if config.barrier_kind not in {RECIPROCAL_BARRIER, LOG_BARRIER}:
        raise ValueError(f"Неизвестный тип барьера: {config.barrier_kind}")

    if not is_strictly_feasible(start_point, problem.constraints, tolerance=config.feasibility_tolerance):
        constraints_values = tuple(constraint.evaluator(start_point) for constraint in problem.constraints)
        raise ValueError(
            "Стартовая точка должна быть строго допустимой: "
            f"g(x) < 0 для всех ограничений, получено {constraints_values}"
        )

    iterations: list[BarrierIterationResult] = []
    last_valid_iteration: BarrierIterationResult | None = None
    failed_iteration: BarrierIterationResult | None = None
    status = BARRIER_STATUS_ERROR
    stop_reason = "Внешний цикл не завершён"
    current_mu = config.mu0
    current_point = start_point

    def build_iteration_result(
        *,
        k: int,
        mu_k: float,
        start_iteration_point: Vector,
        candidate_point: Vector,
        inner_result,
        barrier: float,
        metric: float,
    ) -> BarrierIterationResult:
        objective_value = problem.objective(candidate_point)
        constraints_values = tuple(constraint.evaluator(candidate_point) for constraint in problem.constraints)
        barrier_term = mu_k * barrier
        metric_term = mu_k * metric
        theta_value = objective_value + barrier_term
        return BarrierIterationResult(
            k=k,
            mu_k=mu_k,
            start_point=start_iteration_point,
            x_mu_k=candidate_point,
            objective_value=objective_value,
            barrier_value=barrier,
            barrier_metric=metric,
            barrier_metric_term=metric_term,
            theta_value=theta_value,
            barrier_term=barrier_term,
            constraints_values=constraints_values,
            inner_result=inner_result,
        )

    for k in range(1, config.max_outer_iterations + 1):
        mu_k = current_mu

        def theta(point: Vector) -> float:
            if not is_strictly_feasible(point, problem.constraints, tolerance=config.feasibility_tolerance):
                return math.inf
            barrier = barrier_value(
                point,
                problem.constraints,
                config.barrier_kind,
                tolerance=config.feasibility_tolerance,
            )
            if not math.isfinite(barrier):
                return math.inf
            value = problem.objective(point) + mu_k * barrier
            return value if math.isfinite(value) else math.inf

        inner_result = rosenbrock_minimize(
            objective=theta,
            start_point=current_point,
            config=config.inner_solver_config,
        )
        candidate_point = inner_result.optimum_point
        candidate_feasible = is_strictly_feasible(
            candidate_point,
            problem.constraints,
            tolerance=config.feasibility_tolerance,
        )
        barrier = barrier_value(
            candidate_point,
            problem.constraints,
            config.barrier_kind,
            tolerance=config.feasibility_tolerance,
        )
        metric = barrier_metric(
            candidate_point,
            problem.constraints,
            tolerance=config.feasibility_tolerance,
        )
        candidate_valid = (
            inner_result.success
            and candidate_feasible
            and math.isfinite(barrier)
            and math.isfinite(metric)
            and math.isfinite(problem.objective(candidate_point))
        )
        candidate_iteration = build_iteration_result(
            k=k,
            mu_k=mu_k,
            start_iteration_point=inner_result.start_point,
            candidate_point=candidate_point,
            inner_result=inner_result,
            barrier=barrier,
            metric=metric,
        )
        if not candidate_valid:
            failed_iteration = candidate_iteration
            if last_valid_iteration is None:
                stop_reason = (
                    "Расчёт завершён без валидного допустимого результата: "
                    f"следующий шаг на mu={mu_k:.10g} нарушил допустимость или численный контракт"
                )
                status = BARRIER_STATUS_ERROR
            else:
                stop_reason = (
                    "Расчёт остановлен с предупреждением: "
                    f"следующий шаг на mu={mu_k:.10g} нарушил допустимость или численный контракт"
                )
                status = BARRIER_STATUS_WARNING
            break

        iterations.append(candidate_iteration)
        last_valid_iteration = candidate_iteration
        current_point = candidate_point
        if candidate_iteration.barrier_metric_term < config.epsilon_outer:
            stop_reason = (
                "Внешний цикл завершён по критерию "
                f"mu_k * M(x_mu_k) = {candidate_iteration.barrier_metric_term:.10g} < epsilon_outer"
            )
            status = BARRIER_STATUS_SUCCESS
            break
        if k == config.max_outer_iterations:
            stop_reason = (
                "Внешний цикл остановлен по защитному лимиту внешних итераций "
                f"({config.max_outer_iterations})"
            )
            status = BARRIER_STATUS_WARNING
            break
        current_mu *= config.beta

    if last_valid_iteration is not None:
        optimum_point = last_valid_iteration.x_mu_k
        optimum_value = last_valid_iteration.objective_value
        optimum_constraints = last_valid_iteration.constraints_values
        if status == BARRIER_STATUS_ERROR:
            status = BARRIER_STATUS_WARNING
    else:
        optimum_point = start_point
        optimum_value = problem.objective(start_point)
        optimum_constraints = tuple(constraint.evaluator(start_point) for constraint in problem.constraints)

    return BarrierResult(
        problem=problem,
        start_point=start_point,
        config=config,
        iterations=tuple(iterations),
        last_valid_outer_iteration=last_valid_iteration,
        failed_outer_iteration=failed_iteration,
        optimum_point=optimum_point,
        optimum_value=optimum_value,
        optimum_constraints=optimum_constraints,
        success=status == BARRIER_STATUS_SUCCESS,
        stop_reason=stop_reason,
        status=status,
    )
