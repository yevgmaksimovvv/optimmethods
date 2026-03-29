"""Аналитические helpers для прикладного слоя.

Здесь собраны функции, которые не запускают сами методы поиска, но помогают
объяснить и интерпретировать их результаты:
- выбрать набор методов для расчёта;
- построить разумный диапазон графика;
- вычислить теоретический ориентир;
- сформулировать наблюдения по серии запусков.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from lr1.domain.models import GridRunResult, InputConfig, ReferencePoint, SkippedRun
from lr1.domain.numerical import is_better
from lr1.domain.search import METHOD_ORDER, METHOD_SPECS
from lr1.infrastructure.logging import configure_logging
from lr1.infrastructure.settings import GRID_L_VALUES, SERIES_EPS_VALUES


configure_logging()
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


def _describe_reference_source(source: str) -> str:
    """Преобразует короткий источник ориентира в читабельный русский текст."""
    if source == "левая граница":
        return "на левой границе"
    if source == "правая граница":
        return "на правой границе"
    return source


def _format_count_ru(value: int, one: str, few: str, many: str) -> str:
    """Выбирает правильную форму русского слова по числу."""
    mod10 = value % 10
    mod100 = value % 100
    if mod10 == 1 and mod100 != 11:
        return one
    if 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
        return few
    return many


def build_grid_observations(
    kind: str,
    method_key: str,
    successful_runs: Sequence[GridRunResult],
    skipped_runs: Sequence[SkippedRun],
    reference_point: Optional[ReferencePoint],
) -> Tuple[str, ...]:
    """Формирует наблюдения и выводы для режима серии расчётов.

    Функция пытается не просто перечислить факты, а дать короткие полезные
    интерпретации: где лежит теоретический экстремум, сколько сочетаний
    параметров было пропущено, и как меняется стоимость метода при разных
    значениях `ε` и `l`.
    """
    del kind
    lines: List[str] = []

    if reference_point is not None:
        if reference_point.source in {"левая граница", "правая граница"}:
            lines.append(
                "Экстремум на этом интервале граничный: "
                f"теоретически он достигается {_describe_reference_source(reference_point.source)} "
                f"(x = {reference_point.x:.6f}, f(x) = {reference_point.f:.6f})."
            )
        else:
            lines.append(
                f"Экстремум на этом интервале внутренний: теоретически x = {reference_point.x:.6f}, "
                f"f(x) = {reference_point.f:.6f}."
            )

    if skipped_runs:
        skipped_count = len(skipped_runs)
        combination_word = _format_count_ru(skipped_count, "комбинация", "комбинации", "комбинаций")
        lines.append(
            f"Пропущено {skipped_count} невалидные {combination_word} параметров. "
            "Это не ошибка метода, а ограничение на допустимые значения ε и l."
        )

    if successful_runs:
        best_run = min(successful_runs, key=lambda item: item.result.func_evals)
        lines.append(
            f"Самый экономный запуск по числу вызовов функции: {METHOD_SPECS[best_run.method_key].title} "
            f"при ε={best_run.eps}, l={best_run.l} (вызовов: {best_run.result.func_evals})."
        )

        for current_method_key in resolve_method_keys(method_key):
            method_runs = [item for item in successful_runs if item.method_key == current_method_key]
            if len(method_runs) < 2:
                continue

            avg_by_l: Dict[float, float] = {}
            for l_value in GRID_L_VALUES:
                evals = [item.result.func_evals for item in method_runs if item.l == l_value]
                if evals:
                    avg_by_l[l_value] = sum(evals) / len(evals)
            if len(avg_by_l) == 2 and avg_by_l[min(avg_by_l)] > avg_by_l[max(avg_by_l)]:
                lines.append(
                    f"Для {METHOD_SPECS[current_method_key].title} уменьшение l "
                    f"с {max(avg_by_l):g} до {min(avg_by_l):g} увеличивает среднее число вызовов функции."
                )

            avg_by_eps: Dict[float, float] = {}
            for eps_value in SERIES_EPS_VALUES:
                evals = [item.result.func_evals for item in method_runs if item.eps == eps_value]
                if evals:
                    avg_by_eps[eps_value] = sum(evals) / len(evals)
            if len(avg_by_eps) >= 2:
                eps_min = min(avg_by_eps)
                eps_max = max(avg_by_eps)
                if avg_by_eps[eps_min] > avg_by_eps[eps_max]:
                    lines.append(
                        f"Для {METHOD_SPECS[current_method_key].title} уменьшение ε "
                        f"с {eps_max:g} до {eps_min:g} повышает среднее число вызовов функции."
                    )
                elif abs(avg_by_eps[eps_min] - avg_by_eps[eps_max]) < 1e-9:
                    lines.append(
                        f"Для {METHOD_SPECS[current_method_key].title} в этой сетке число вызовов функции почти не зависит от ε."
                    )

    return tuple(lines)
