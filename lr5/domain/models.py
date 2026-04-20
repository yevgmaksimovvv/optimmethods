"""Контракты доменного слоя ЛР5."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lr2.domain.models import SolverConfig, SolverResult, Vector

BarrierValue = float

BARRIER_STATUS_SUCCESS = "success"
BARRIER_STATUS_WARNING = "warning"
BARRIER_STATUS_ERROR = "error"


@dataclass(frozen=True)
class BarrierConstraint:
    """Ограничение вида `g(x) <= 0`."""

    title: str
    evaluator: Callable[[Vector], float]


@dataclass(frozen=True)
class BarrierProblem:
    """Постановка задачи условной оптимизации для барьерного метода."""

    title: str
    objective_title: str
    objective: Callable[[Vector], float]
    constraints: tuple[BarrierConstraint, ...]


@dataclass(frozen=True)
class BarrierMethodConfig:
    """Параметры барьерного метода."""

    mu0: float
    beta: float
    epsilon_outer: float
    max_outer_iterations: int
    barrier_kind: str
    inner_solver_config: SolverConfig
    feasibility_tolerance: float = 0.0


@dataclass(frozen=True)
class BarrierIterationResult:
    """Результат одной внешней итерации барьерного метода."""

    k: int
    mu_k: float
    start_point: Vector
    x_mu_k: Vector
    objective_value: float
    barrier_value: float
    barrier_metric: float
    barrier_metric_term: float
    theta_value: float
    barrier_term: float
    constraints_values: tuple[float, ...]
    inner_result: SolverResult


@dataclass(frozen=True)
class BarrierResult:
    """Итог работы барьерного метода."""

    problem: BarrierProblem
    start_point: Vector
    config: BarrierMethodConfig
    iterations: tuple[BarrierIterationResult, ...]
    last_valid_outer_iteration: BarrierIterationResult | None
    failed_outer_iteration: BarrierIterationResult | None
    optimum_point: Vector
    optimum_value: float
    optimum_constraints: tuple[float, ...]
    success: bool
    stop_reason: str
    status: str = BARRIER_STATUS_ERROR
