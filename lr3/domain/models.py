"""Контракты доменного слоя ЛР3."""

from __future__ import annotations

from dataclasses import dataclass

Point2D = tuple[float, float]


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
