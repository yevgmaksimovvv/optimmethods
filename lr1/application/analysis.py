"""Аналитические helpers для прикладного слоя.

Здесь собраны функции, которые не запускают сами методы поиска, но помогают
объяснить и интерпретировать их результаты:
- выбрать набор методов для расчёта;
- построить разумный диапазон графика;
- вычислить теоретический ориентир;
"""

import logging
from typing import List, Optional, Tuple

from lr1.domain.models import InputConfig, ReferencePoint
from lr1.domain.numerical import is_better
from lr1.domain.search import METHOD_ORDER

logger = logging.getLogger("lr1.analysis")

def resolve_method_keys(method_key: str) -> Tuple[str, ...]:
    """Нормализует выбор метода из UI в кортеж ключей.

    Это упрощает остальной код: сервису не нужно отдельно ветвиться
    на случай `all` и одиночного метода.
    """
    logger.debug("resolve_method_keys input=%s", method_key)
    if method_key == "all":
        return METHOD_ORDER
    return (method_key,)


def build_plot_range(function_spec, interval_raw: Tuple[float, float]) -> Tuple[float, float]:
    """Расширяет пользовательский интервал до более удобного диапазона графика.

    На графике полезно видеть не только рабочий отрезок `[a, b]`, но и немного
    контекста вокруг него. Для рациональных функций диапазон дополнительно
    растягивается так, чтобы в обзор попадали точки разрыва.
    """
    logger.debug("build_plot_range function=%s interval_raw=%s", function_spec.key, interval_raw)
    a, b = interval_raw
    margin = max(1.0, 0.2 * (b - a))
    left = a - margin
    right = b + margin

    if function_spec.forbidden_points:
        left = min(left, min(function_spec.forbidden_points) - margin)
        right = max(right, max(function_spec.forbidden_points) + margin)

    plot_range = (left, right)
    logger.debug("build_plot_range result=%s", plot_range)
    return plot_range


def is_valid_method_params(method_key: str, eps: float, interval_l: float) -> bool:
    """Проверяет, допустима ли пара параметров для конкретного метода."""
    if method_key == "dichotomy":
        return eps < interval_l
    return True


def skip_reason(method_key: str) -> str:
    """Возвращает текстовое объяснение, почему прогон был пропущен."""
    if method_key == "dichotomy":
        return "для метода дихотомии нужно ε < l"
    return "невалидная комбинация параметров"


def theoretical_optimum(config: InputConfig) -> Optional[ReferencePoint]:
    """Ищет теоретический ориентир среди границ и стационарных точек.

    Это не строгий символьный решатель в общем виде, а учебный ориентир,
    который помогает сравнить численный результат с ожидаемым кандидатом
    на экстремум внутри заданного интервала.
    """
    a, b = config.interval
    candidates: List[ReferencePoint] = []

    for x, source in ((a, "левая граница"), (b, "правая граница")):
        try:
            candidates.append(ReferencePoint(x=x, f=config.function_spec.func(x), source=source))
        except ZeroDivisionError:
            logger.debug("Skipped theoretical candidate x=%s due to undefined function", x)

    for x_value in config.function_spec.stationary_points:
        if a < x_value < b:
            try:
                candidates.append(
                    ReferencePoint(x=x_value, f=config.function_spec.func(x_value), source="стационарная точка")
                )
            except ZeroDivisionError:
                logger.debug("Skipped stationary point x=%s due to undefined function", x_value)

    if not candidates:
        return None

    best = candidates[0]
    for candidate in candidates[1:]:
        if is_better(candidate.f, best.f, config.kind):
            best = candidate
    return best
