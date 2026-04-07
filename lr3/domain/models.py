"""Контракты доменного слоя ЛР3."""

from __future__ import annotations

from dataclasses import dataclass

Point2D = tuple[float, float]
Hessian2D = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class MethodConfig:
    """Параметры численных методов ЛР3."""

    epsilon: float
    max_iterations: int
    initial_step: float
    min_step: float
    timeout_seconds: float
    gradient_step: float
    max_step_expansions: int


@dataclass(frozen=True)
class IterationRecord:
    """Одна строка итерационного процесса."""

    k: int
    point: Point2D
    value: float
    gradient: Point2D
    step_size: float
    direction: Point2D | None = None
    beta: float | None = None
    next_point: Point2D | None = None
    next_value: float | None = None
    restart_direction: bool | None = None


@dataclass(frozen=True)
class OptimizationResult:
    """Итог запуска метода оптимизации."""

    method_name: str
    start_point: Point2D
    optimum_point: Point2D
    optimum_value: float
    iterations_count: int
    records: tuple[IterationRecord, ...]
    success: bool
    stop_reason: str


@dataclass(frozen=True)
class ExtremumAnalysis:
    """Аналитический разбор функции до численного метода."""

    expression: str
    start_point: Point2D
    goal: str
    gradient_formula: tuple[str, str]
    gradient_at_start: Point2D
    hessian_formula: Hessian2D
    stationary_points: tuple[Point2D, ...]
    stationary_gradient: Point2D | None
    hessian_at_stationary_point: Hessian2D | None
    stationary_point_kind: str
    theory_conclusion: str
    goal_alignment: str
    strictness_note: str
