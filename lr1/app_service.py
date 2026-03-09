import logging
import math
import time
from typing import Dict, List, Optional, Tuple

if __package__:
    from .app_models import GridRunResult, InputConfig, RunReport, SearchResult, SkippedRun
    from .function_defs import FUNCTION_SPECS
    from .logging_setup import configure_logging
    from .search_methods import METHOD_ORDER, METHOD_SPECS, analytic_comment, sanitize_interval
else:
    from app_models import GridRunResult, InputConfig, RunReport, SearchResult, SkippedRun
    from function_defs import FUNCTION_SPECS
    from logging_setup import configure_logging
    from search_methods import METHOD_ORDER, METHOD_SPECS, analytic_comment, sanitize_interval


configure_logging()
logger = logging.getLogger("lr1.app_service")

GRID_EPS_VALUES = (0.1, 0.01, 0.001)
GRID_L_VALUES = (0.1, 0.01)


def _better(v1: float, v2: float, kind: str) -> bool:
    if kind == "max":
        return v1 > v2
    if kind == "min":
        return v1 < v2
    raise ValueError(f"Неизвестный тип поиска: {kind}")


def _parse_float(raw_value: str, field_name: str) -> float:
    logger.debug("Parsing float field=%s raw_value=%s", field_name, raw_value)
    try:
        return float(raw_value.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Неверное значение для {field_name}: {raw_value}") from exc


def build_input_config(
    function_key: str,
    kind: str,
    method_key: str,
    a_raw: str,
    b_raw: str,
    eps_raw: str,
    l_raw: str,
) -> InputConfig:
    logger.info(
        "build_input_config function=%s kind=%s method=%s a=%s b=%s eps=%s l=%s",
        function_key,
        kind,
        method_key,
        a_raw,
        b_raw,
        eps_raw,
        l_raw,
    )
    if function_key not in FUNCTION_SPECS:
        raise ValueError(f"Неизвестная функция: {function_key}")
    if method_key != "all" and method_key not in METHOD_SPECS:
        raise ValueError(f"Неизвестный метод: {method_key}")
    if kind not in {"max", "min"}:
        raise ValueError(f"Неизвестный тип поиска: {kind}")

    a = _parse_float(a_raw, "a")
    b = _parse_float(b_raw, "b")
    eps = _parse_float(eps_raw, "eps")
    l = _parse_float(l_raw, "l")

    if a >= b:
        raise ValueError("Должно быть a < b.")
    if eps <= 0 or l <= 0:
        raise ValueError("eps и l должны быть положительными.")

    function_spec = FUNCTION_SPECS[function_key]
    shift = max(1e-10, 1e-8 * max(1.0, abs(a), abs(b)))
    interval = sanitize_interval(a, b, forbidden_points=list(function_spec.forbidden_points), shift=shift)

    config = InputConfig(
        function_spec=function_spec,
        kind=kind,
        method_key=method_key,
        interval_raw=(a, b),
        interval=interval,
        eps=eps,
        l=l,
    )
    logger.info("build_input_config success function=%s interval=%s", config.function_spec.key, config.interval)
    return config


def resolve_method_keys(method_key: str) -> Tuple[str, ...]:
    logger.debug("resolve_method_keys input=%s", method_key)
    if method_key == "all":
        return METHOD_ORDER
    return (method_key,)


def build_plot_range(function_key: str, interval_raw: Tuple[float, float]) -> Tuple[float, float]:
    logger.debug("build_plot_range function=%s interval_raw=%s", function_key, interval_raw)
    a, b = interval_raw
    margin = max(1.0, 0.2 * (b - a))
    left = a - margin
    right = b + margin

    if function_key == "F2":
        left = min(left, -10.0)
        right = max(right, 10.0)

    plot_range = (left, right)
    logger.debug("build_plot_range result=%s", plot_range)
    return plot_range


def _result_block(result: SearchResult) -> List[str]:
    return [
        f"[{result.method}]",
        f"x* = {result.x_opt:.10f}",
        f"f(x*) = {result.f_opt:.10f}",
        f"Итераций = {len(result.iterations)}",
        f"Вычислений функции = {result.func_evals}",
        f"Финальный интервал = [{result.interval_final[0]:.10f}, {result.interval_final[1]:.10f}]",
        "",
    ]


def _is_valid_method_params(method_key: str, eps: float, l: float) -> bool:
    if method_key == "dichotomy":
        return eps < l
    return True


def _skip_reason(method_key: str) -> str:
    if method_key == "dichotomy":
        return "для дихотомии нужно eps < l"
    return "невалидная комбинация параметров"


def _describe_reference_source(source: str) -> str:
    if source == "левая граница":
        return "на левой границе"
    if source == "правая граница":
        return "на правой границе"
    return source


def _format_count_ru(value: int, one: str, few: str, many: str) -> str:
    mod10 = value % 10
    mod100 = value % 100
    if mod10 == 1 and mod100 != 11:
        return one
    if 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
        return few
    return many


def _theoretical_optimum(
    config: InputConfig,
) -> Optional[Tuple[float, float, str]]:
    a, b = config.interval
    candidates: List[Tuple[float, float, str]] = []

    for x, source in ((a, "левая граница"), (b, "правая граница")):
        try:
            candidates.append((x, config.function_spec.func(x), source))
        except ZeroDivisionError:
            logger.debug("Skipped theoretical candidate x=%s due to undefined function", x)

    if config.function_spec.key == "F1":
        x_vertex = 2.5
        if a <= x_vertex <= b:
            candidates.append((x_vertex, config.function_spec.func(x_vertex), "вершина параболы"))
    elif config.function_spec.key == "F2":
        stationary_points = (
            ((19.0 - math.sqrt(385.0)) / 4.0, "стационарная точка на ветви (-4, 2)"),
            ((19.0 + math.sqrt(385.0)) / 4.0, "стационарная точка на ветви (2, +inf)"),
        )
        for x_value, source in stationary_points:
            if a < x_value < b:
                try:
                    candidates.append((x_value, config.function_spec.func(x_value), source))
                except ZeroDivisionError:
                    logger.debug("Skipped F2 stationary point x=%s due to undefined function", x_value)

    if not candidates:
        return None

    best_x, best_f, best_source = candidates[0]
    for x_value, f_value, source in candidates[1:]:
        if _better(f_value, best_f, config.kind):
            best_x, best_f, best_source = x_value, f_value, source
    return best_x, best_f, best_source


def _build_reference_lines(
    config: InputConfig,
    results_by_method: Dict[str, SearchResult],
) -> List[str]:
    lines = [
        "Теоретическая справка:",
        analytic_comment(config.function_spec.key, config.interval, config.kind) or "—",
    ]
    reference = _theoretical_optimum(config)
    if reference is None:
        return lines

    x_ref, f_ref, source = reference
    lines.extend(
        [
            "",
            "Теоретический ориентир на этом интервале:",
            f"x* = {x_ref:.10f}",
            f"f(x*) = {f_ref:.10f}",
            f"Источник: {source}",
        ]
    )
    if results_by_method:
        lines.append("")
        lines.append("Отклонение найденных решений:")
        for method_key, result in results_by_method.items():
            lines.append(
                f"{METHOD_SPECS[method_key].title}: "
                f"|dx| = {abs(result.x_opt - x_ref):.10f}, "
                f"|df| = {abs(result.f_opt - f_ref):.10f}"
            )
    return lines


def _grid_observation_lines(
    config: InputConfig,
    successful_runs: List[Tuple[str, float, float, SearchResult]],
    skipped_runs: List[Tuple[str, float, float, str]],
) -> List[str]:
    lines: List[str] = []
    reference = _theoretical_optimum(config)

    if reference is not None:
        x_ref, f_ref, source = reference
        if source.startswith("левая граница") or source.startswith("правая граница"):
            lines.append(
                f"Экстремум на этом интервале граничный: теоретически он достигается {_describe_reference_source(source)} "
                f"(x = {x_ref:.6f}, f(x) = {f_ref:.6f})."
            )
        else:
            lines.append(
                f"Экстремум на этом интервале внутренний: теоретически x = {x_ref:.6f}, f(x) = {f_ref:.6f}."
            )

    if skipped_runs:
        skipped_count = len(skipped_runs)
        combination_word = _format_count_ru(skipped_count, "комбинация", "комбинации", "комбинаций")
        lines.append(
            f"Пропущено {skipped_count} невалидные {combination_word} параметров. "
            "Это не ошибка метода, а ограничение на допустимые eps и l."
        )

    if successful_runs:
        best_run = min(successful_runs, key=lambda item: item[3].func_evals)
        lines.append(
            f"Самый экономный запуск по числу вычислений функции: {METHOD_SPECS[best_run[0]].title} "
            f"при eps={best_run[1]}, l={best_run[2]} (evals={best_run[3].func_evals})."
        )

        for method_key in resolve_method_keys(config.method_key):
            method_runs = [item for item in successful_runs if item[0] == method_key]
            if len(method_runs) < 2:
                continue

            avg_by_l: Dict[float, float] = {}
            for l_value in GRID_L_VALUES:
                evals = [item[3].func_evals for item in method_runs if item[2] == l_value]
                if evals:
                    avg_by_l[l_value] = sum(evals) / len(evals)
            if len(avg_by_l) == 2 and avg_by_l[min(avg_by_l)] > avg_by_l[max(avg_by_l)]:
                lines.append(
                    f"Для {METHOD_SPECS[method_key].title} уменьшение l "
                    f"с {max(avg_by_l):g} до {min(avg_by_l):g} увеличивает среднее число evals."
                )

            avg_by_eps: Dict[float, float] = {}
            for eps_value in GRID_EPS_VALUES:
                evals = [item[3].func_evals for item in method_runs if item[1] == eps_value]
                if evals:
                    avg_by_eps[eps_value] = sum(evals) / len(evals)
            if len(avg_by_eps) >= 2:
                eps_min = min(avg_by_eps)
                eps_max = max(avg_by_eps)
                if avg_by_eps[eps_min] > avg_by_eps[eps_max]:
                    lines.append(
                        f"Для {METHOD_SPECS[method_key].title} уменьшение eps "
                        f"с {eps_max:g} до {eps_min:g} повышает среднее число evals."
                    )
                elif abs(avg_by_eps[eps_min] - avg_by_eps[eps_max]) < 1e-9:
                    lines.append(
                        f"Для {METHOD_SPECS[method_key].title} в этой сетке число evals практически не зависит от eps."
                    )

    return lines


def run_single(config: InputConfig) -> RunReport:
    started = time.perf_counter()
    logger.info(
        "run_single start function=%s kind=%s method=%s interval=%s eps=%s l=%s",
        config.function_spec.key,
        config.kind,
        config.method_key,
        config.interval,
        config.eps,
        config.l,
    )
    method_keys = resolve_method_keys(config.method_key)
    results_by_method: Dict[str, SearchResult] = {}

    lines = [
        f"Функция: {config.function_spec.title}",
        f"Исходный интервал: [{config.interval_raw[0]}, {config.interval_raw[1]}]",
    ]
    if config.interval != config.interval_raw:
        lines.append(
            f"Скорректированный интервал для вычислений: [{config.interval[0]:.10f}, {config.interval[1]:.10f}]"
        )
    lines.extend(
        [
            f"Поиск: {config.kind}",
            f"eps = {config.eps}, l = {config.l}",
            "",
        ]
    )

    for method_key in method_keys:
        try:
            result = METHOD_SPECS[method_key].runner(
                config.function_spec.func,
                config.interval[0],
                config.interval[1],
                config.eps,
                config.l,
                config.kind,
            )
        except ValueError as method_error:
            logger.warning("run_single method=%s failed error=%s", method_key, method_error)
            lines.append(f"{METHOD_SPECS[method_key].title}: ошибка -> {method_error}")
            lines.append("")
            continue

        results_by_method[method_key] = result
        logger.info(
            "run_single method=%s success x_opt=%.10f f_opt=%.10f iterations=%d evals=%d",
            method_key,
            result.x_opt,
            result.f_opt,
            len(result.iterations),
            result.func_evals,
        )
        lines.extend(_result_block(result))

    if not results_by_method:
        raise ValueError("Ни один метод не смог выполнить расчёт с текущими параметрами.")

    analytic_note = analytic_comment(config.function_spec.key, config.interval, config.kind) or "—"
    reference = _theoretical_optimum(config)
    lines.extend(_build_reference_lines(config, results_by_method))

    report = RunReport(
        summary_text="\n".join(lines),
        results_by_method=results_by_method,
        method_keys=method_keys,
        default_method_key=next(iter(results_by_method), None),
        function_spec=config.function_spec,
        kind=config.kind,
        interval_raw=config.interval_raw,
        interval=config.interval,
        eps=config.eps,
        l=config.l,
        plot_range=build_plot_range(config.function_spec.key, config.interval_raw),
        analytic_note=analytic_note,
        reference_x=reference[0] if reference is not None else None,
        reference_f=reference[1] if reference is not None else None,
        reference_source=reference[2] if reference is not None else "",
        mode="single",
    )
    logger.info(
        "run_single done results=%d duration_ms=%.2f",
        len(results_by_method),
        (time.perf_counter() - started) * 1000.0,
    )
    return report


def run_full_grid(config: InputConfig) -> RunReport:
    started = time.perf_counter()
    logger.info(
        "run_full_grid start function=%s kind=%s method=%s interval=%s",
        config.function_spec.key,
        config.kind,
        config.method_key,
        config.interval,
    )
    method_keys = resolve_method_keys(config.method_key)
    best_by_method: Dict[str, List[Tuple[float, float, SearchResult]]] = {key: [] for key in method_keys}
    successful_runs: List[Tuple[str, float, float, SearchResult]] = []
    skipped_runs: List[Tuple[str, float, float, str]] = []
    grid_runs_by_method: Dict[str, List[GridRunResult]] = {key: [] for key in method_keys}

    lines = [
        "Полный перебор параметров",
        f"Функция: {config.function_spec.title}",
        f"Интервал: [{config.interval_raw[0]}, {config.interval_raw[1]}]",
    ]
    if config.interval != config.interval_raw:
        lines.append(f"Скорректированный интервал: [{config.interval[0]:.10f}, {config.interval[1]:.10f}]")
    lines.extend(
        [
            f"Поиск: {config.kind}",
            "",
        ]
    )

    for eps in GRID_EPS_VALUES:
        for l in GRID_L_VALUES:
            logger.debug("run_full_grid combination eps=%s l=%s", eps, l)
            lines.append(f"=== eps={eps}, l={l} ===")
            for method_key in method_keys:
                if not _is_valid_method_params(method_key, eps, l):
                    reason = _skip_reason(method_key)
                    logger.info(
                        "run_full_grid method=%s skipped eps=%s l=%s reason=%s",
                        method_key,
                        eps,
                        l,
                        reason,
                    )
                    skipped_runs.append((method_key, eps, l, reason))
                    lines.append(f"{METHOD_SPECS[method_key].title}: пропущено -> {reason}")
                    continue
                try:
                    result = METHOD_SPECS[method_key].runner(
                        config.function_spec.func,
                        config.interval[0],
                        config.interval[1],
                        eps,
                        l,
                        config.kind,
                    )
                except ValueError as method_error:
                    logger.warning(
                        "run_full_grid method=%s failed eps=%s l=%s error=%s",
                        method_key,
                        eps,
                        l,
                        method_error,
                    )
                    lines.append(f"{METHOD_SPECS[method_key].title}: ошибка -> {method_error}")
                    continue

                best_by_method[method_key].append((eps, l, result))
                successful_runs.append((method_key, eps, l, result))
                grid_runs_by_method[method_key].append(
                    GridRunResult(method_key=method_key, eps=eps, l=l, result=result)
                )
                logger.debug(
                    "run_full_grid method=%s success eps=%s l=%s x_opt=%.10f f_opt=%.10f evals=%d",
                    method_key,
                    eps,
                    l,
                    result.x_opt,
                    result.f_opt,
                    result.func_evals,
                )
                lines.append(
                    f"{result.method}: x*={result.x_opt:.8f}, "
                    f"f(x*)={result.f_opt:.8f}, "
                    f"iter={len(result.iterations)}, evals={result.func_evals}"
                )
            lines.append("")

    lines.append("Сравнение по числу вычислений функции:")
    results_by_method: Dict[str, SearchResult] = {}
    for method_key in method_keys:
        runs = best_by_method[method_key]
        if not runs:
            continue

        best_run = min(runs, key=lambda item: item[2].func_evals)
        worst_run = max(runs, key=lambda item: item[2].func_evals)
        lines.append(
            f"{METHOD_SPECS[method_key].title}: "
            f"минимум evals = {best_run[2].func_evals} при eps={best_run[0]}, l={best_run[1]}; "
            f"максимум evals = {worst_run[2].func_evals} при eps={worst_run[0]}, l={worst_run[1]}"
        )
        results_by_method[method_key] = best_run[2]

    reference = _theoretical_optimum(config)
    if reference is not None:
        x_ref, f_ref, source = reference
        lines.extend(
            [
                "",
                "Теоретический ориентир на этом интервале:",
                f"x* = {x_ref:.10f}",
                f"f(x*) = {f_ref:.10f}",
                f"Источник: {source}",
            ]
        )
        if results_by_method:
            lines.append("")
            lines.append("Отклонение лучших запусков:")
            for method_key, result in results_by_method.items():
                lines.append(
                    f"{METHOD_SPECS[method_key].title}: "
                    f"|dx| = {abs(result.x_opt - x_ref):.10f}, "
                    f"|df| = {abs(result.f_opt - f_ref):.10f}"
                )

    observation_lines = tuple(_grid_observation_lines(config, successful_runs, skipped_runs))
    lines.extend(["", "Краткий вывод:"])
    lines.extend(observation_lines)

    default_method_key = next(iter(results_by_method), None)
    available_method_keys = tuple(key for key in method_keys if key in results_by_method)
    report = RunReport(
        summary_text="\n".join(lines),
        results_by_method=results_by_method,
        method_keys=available_method_keys,
        default_method_key=default_method_key,
        function_spec=config.function_spec,
        kind=config.kind,
        interval_raw=config.interval_raw,
        interval=config.interval,
        eps=None,
        l=None,
        plot_range=build_plot_range(config.function_spec.key, config.interval_raw),
        analytic_note=analytic_comment(config.function_spec.key, config.interval, config.kind) or "—",
        reference_x=reference[0] if reference is not None else None,
        reference_f=reference[1] if reference is not None else None,
        reference_source=reference[2] if reference is not None else "",
        observations=observation_lines,
        mode="grid",
        grid_runs_by_method={key: tuple(grid_runs_by_method[key]) for key in available_method_keys},
        skipped_runs=tuple(
            SkippedRun(method_key=method_key, eps=eps, l=l, reason=reason)
            for method_key, eps, l, reason in skipped_runs
        ),
    )
    logger.info(
        "run_full_grid done result_methods=%s duration_ms=%.2f",
        available_method_keys,
        (time.perf_counter() - started) * 1000.0,
    )
    return report
