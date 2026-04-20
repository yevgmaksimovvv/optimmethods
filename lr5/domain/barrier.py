"""Барьерные функции и проверки допустимости."""

from __future__ import annotations

import math
from collections.abc import Iterable

from lr2.domain.models import Vector
from lr5.domain.models import BarrierConstraint

BarrierKind = str
RECIPROCAL_BARRIER = "reciprocal"
LOG_BARRIER = "log"


def evaluate_constraints(point: Vector, constraints: Iterable[BarrierConstraint]) -> tuple[float, ...]:
    return tuple(constraint.evaluator(point) for constraint in constraints)


def is_strictly_feasible(
    point: Vector,
    constraints: Iterable[BarrierConstraint],
    *,
    tolerance: float = 0.0,
) -> bool:
    return all(value < -tolerance for value in evaluate_constraints(point, constraints))


def barrier_value(
    point: Vector,
    constraints: Iterable[BarrierConstraint],
    barrier_kind: BarrierKind,
    *,
    tolerance: float = 0.0,
) -> float:
    values = evaluate_constraints(point, constraints)
    if any(value >= -tolerance for value in values):
        return math.inf

    if barrier_kind == RECIPROCAL_BARRIER:
        return sum(-1.0 / value for value in values)
    if barrier_kind == LOG_BARRIER:
        return -sum(math.log(-value) for value in values)
    raise ValueError(f"Неизвестный тип барьера: {barrier_kind}")


def barrier_metric(
    point: Vector,
    constraints: Iterable[BarrierConstraint],
    *,
    tolerance: float = 0.0,
) -> float:
    """Возвращает положительную метрику близости к границе для внешнего останова.

    Метрика не зависит от вида барьера. Для строго допустимой точки это сумма
    обратных запасов `1 / (-g_i(x))`. Она совпадает по смыслу с методической
    схемой для reciprocal barrier и остаётся знакопостоянной для log barrier.
    """

    values = evaluate_constraints(point, constraints)
    if any(value >= -tolerance for value in values):
        return math.inf
    return sum(-1.0 / value for value in values)
