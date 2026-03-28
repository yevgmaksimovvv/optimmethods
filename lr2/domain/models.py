"""Контракты доменного слоя для ЛР2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Vector = tuple[float, ...]


@dataclass(frozen=True)
class Polynomial2D:
    """Полином двух переменных: sum(c[i][j] * x1^i * x2^j)."""

    coefficients: tuple[tuple[float, ...], ...]
    title: str

    @property
    def degree_x1(self) -> int:
        return len(self.coefficients) - 1

    @property
    def degree_x2(self) -> int:
        return len(self.coefficients[0]) - 1 if self.coefficients else -1


@dataclass(frozen=True)
class SolverConfig:
    """Входные параметры метода Розенброка."""

    epsilon: float
    max_iterations: int
    line_search_initial_step: float
    line_search_growth: float
    line_search_max_expand: int
    line_search_samples: int
    line_search_tolerance: float
    line_search_max_iterations: int
    direction_zero_tolerance: float
    stagnation_abs_tolerance: float
    stagnation_rel_tolerance: float


@dataclass(frozen=True)
class IterationStep:
    """Одна строка таблицы итераций по заданию."""

    k: int
    x_k: Vector
    f_x_k: float
    j: int
    direction: Vector
    y_j: Vector
    f_y_j: float
    lambda_j: float
    y_next: Vector
    f_y_next: float


@dataclass(frozen=True)
class SolverResult:
    """Итог одного прогона метода."""

    epsilon: float
    start_point: Vector
    optimum_point: Vector
    optimum_value: float
    iterations_count: int
    steps: tuple[IterationStep, ...]
    trajectory: tuple[Vector, ...]
    success: bool
    stop_reason: str


@dataclass(frozen=True)
class BatchResult:
    """Результат серии запусков для нескольких eps и стартовых точек."""

    polynomial: Polynomial2D
    runs: tuple[SolverResult, ...]


def normalize_coefficients(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Проверяет и нормализует матрицу коэффициентов в кортежи."""
    if not matrix:
        raise ValueError("Матрица коэффициентов пуста.")
    width = len(matrix[0])
    if width == 0:
        raise ValueError("Матрица коэффициентов содержит пустую строку.")
    for row in matrix:
        if len(row) != width:
            raise ValueError("Матрица коэффициентов должна быть прямоугольной.")
    return tuple(tuple(float(value) for value in row) for row in matrix)
