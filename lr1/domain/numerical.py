"""Общие численные утилиты для доменного слоя.

Здесь нет конкретных методов поиска, только маленькие операции,
которые переиспользуются в нескольких алгоритмах:
- сравнение для минимума/максимума;
- выбор "лучшей" стороны интервала;
- безопасная обработка интервала возле точек разрыва;
- работа с наборами чисел с учётом допусков.
"""

import logging
from typing import Iterable, Optional, Sequence, Tuple

from lr1.infrastructure.settings import (
    INTERVAL_SHIFT_BASE,
    MIN_INTERVAL_SHIFT,
    ROOT_TOLERANCE,
)

logger = logging.getLogger("lr1.numerical")


def is_better(v1: float, v2: float, kind: str) -> bool:
    """Возвращает `True`, если `v1` лучше `v2` для заданной цели поиска."""
    if kind == "max":
        return v1 > v2
    if kind == "min":
        return v1 < v2
    raise ValueError("Тип поиска должен быть 'max' или 'min'")


def choose_side(left_value: float, right_value: float, kind: str) -> str:
    """Определяет, в какую сторону нужно сдвигать интервал.

    Возвращает:
    - `"right"`, если правая пробная точка лучше левой;
    - `"left"` в противном случае.
    """
    return "right" if is_better(right_value, left_value, kind) else "left"


def scaled_interval_shift(a: float, b: float) -> float:
    """Подбирает безопасный сдвиг границы интервала в зависимости от масштаба."""
    return max(MIN_INTERVAL_SHIFT, INTERVAL_SHIFT_BASE * max(1.0, abs(a), abs(b)))


def sanitize_interval(
    a: float,
    b: float,
    forbidden_points: Optional[Sequence[float]] = None,
    shift: float = 1e-8,
) -> Tuple[float, float]:
    """Проверяет и при необходимости слегка корректирует интервал.

    Функция нужна для работы с рациональными функциями, у которых есть точки
    разрыва. Если граница интервала совпадает с такой точкой, граница
    аккуратно сдвигается внутрь.
    """
    logger.debug(
        "sanitize_interval start a=%s b=%s forbidden_points=%s shift=%s",
        a,
        b,
        forbidden_points,
        shift,
    )
    if a >= b:
        raise ValueError(f"Invalid interval [{a}, {b}]")

    for point in forbidden_points or ():
        if abs(a - point) < ROOT_TOLERANCE:
            a += shift
        if abs(b - point) < ROOT_TOLERANCE:
            b -= shift

    if a >= b:
        raise ValueError("Интервал схлопнулся после сдвига границ.")

    logger.debug("sanitize_interval result a=%s b=%s", a, b)
    return a, b


def unique_sorted(values: Iterable[float], tol: float = ROOT_TOLERANCE) -> Tuple[float, ...]:
    """Сортирует значения и убирает почти совпадающие элементы."""
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(result[-1] - value) > tol:
            result.append(value)
    return tuple(result)


def far_from_all(value: float, points: Sequence[float], tol: float = ROOT_TOLERANCE) -> bool:
    """Проверяет, что число не совпадает ни с одной точкой из списка."""
    return all(abs(value - point) > tol for point in points)
