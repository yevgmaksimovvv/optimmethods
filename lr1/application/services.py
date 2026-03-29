"""Прикладной слой, который собирает сценарии расчёта.

Этот модуль не реализует сами методы поиска и не рисует интерфейс.
Его задача другая:
- принять сырой пользовательский ввод;
- превратить его в строгий `InputConfig`;
- запустить нужный use case: одиночный расчёт или серию запусков;
- собрать единый `RunReport`, который уже можно безопасно показывать в UI.
"""

import logging
import time
from typing import Dict

from optim_core.parsing import parse_localized_float

from lr1.application.analysis import (
    build_plot_range,
    is_valid_method_params,
    resolve_method_keys,
    skip_reason,
    theoretical_optimum,
)
from lr1.domain.functions import FUNCTION_TEMPLATE_SPECS, build_function_spec
from lr1.domain.models import (
    GridRunResult,
    InputConfig,
    RunReport,
    SearchResult,
    SkippedRun,
)
from lr1.domain.numerical import sanitize_interval, scaled_interval_shift
from lr1.domain.search import METHOD_SPECS
from lr1.infrastructure.logging import configure_logging

configure_logging()
logger = logging.getLogger("lr1.app_service")


def parse_positive_series(raw: str, field_name: str) -> tuple[float, ...]:
    """Парсит список положительных чисел через запятую."""
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise ValueError(f"Список {field_name} пуст.")
    values = tuple(parse_localized_float(item, field_name) for item in parts)
    if any(item <= 0 for item in values):
        raise ValueError(f"Все значения {field_name} должны быть > 0.")
    return values


def build_input_config(
    function_key: str,
    kind: str,
    method_key: str,
    a_raw: str,
    b_raw: str,
    eps_raw: str,
    l_raw: str,
    coefficient_raws: Dict[str, str],
) -> InputConfig:
    """Собирает и валидирует все входные параметры одного расчёта.

    На этом шаге происходит вся подготовка данных перед запуском алгоритмов:
    - проверяется существование выбранной функции и метода;
    - читаются числовые поля интерфейса;
    - строится `FunctionSpec` по коэффициентам;
    - интервал очищается от недопустимых точек разрыва.

    Возвращается полностью готовый `InputConfig`, который уже можно
    без дополнительных проверок передавать в `run_batch`.
    """
    logger.info(
        "build_input_config function=%s kind=%s method=%s a=%s b=%s eps=%s l=%s coefficients=%s",
        function_key,
        kind,
        method_key,
        a_raw,
        b_raw,
        eps_raw,
        l_raw,
        coefficient_raws,
    )
    if function_key not in FUNCTION_TEMPLATE_SPECS:
        raise ValueError(f"Неизвестная функция: {function_key}")
    if method_key != "all" and method_key not in METHOD_SPECS:
        raise ValueError(f"Неизвестный метод: {method_key}")
    if kind not in {"max", "min"}:
        raise ValueError(f"Неизвестный тип поиска: {kind}")

    a = parse_localized_float(a_raw, "a")
    b = parse_localized_float(b_raw, "b")
    eps = parse_localized_float(eps_raw, "ε")
    l_value = parse_localized_float(l_raw, "l")

    if a >= b:
        raise ValueError("Должно быть a < b.")
    if eps <= 0 or l_value <= 0:
        raise ValueError("Параметры ε и l должны быть положительными.")

    template = FUNCTION_TEMPLATE_SPECS[function_key]
    coefficients = {
        item.key: parse_localized_float(coefficient_raws.get(item.key, str(item.default)), item.label)
        for item in template.coefficients
    }
    function_spec = build_function_spec(function_key, coefficients)
    interval = sanitize_interval(
        a,
        b,
        forbidden_points=function_spec.forbidden_points,
        shift=scaled_interval_shift(a, b),
    )

    config = InputConfig(
        function_spec=function_spec,
        kind=kind,
        method_key=method_key,
        interval_raw=(a, b),
        interval=interval,
        eps=eps,
        l=l_value,
    )
    logger.info("build_input_config success function=%s interval=%s", config.function_spec.key, config.interval)
    return config


def run_batch(config: InputConfig, eps_values: tuple[float, ...], l_values: tuple[float, ...]) -> RunReport:
    """Запускает расчёт по всем комбинациям `ε × l` для выбранных методов.

    Контракт единый для двух сценариев:
    - одиночный запуск: `eps_values=(config.eps,)`, `l_values=(config.l,)`;
    - серия запусков: список значений берётся из конфигурационной сетки.
    """
    started = time.perf_counter()
    logger.info(
        "run_batch start function=%s kind=%s method=%s interval=%s eps_values=%s l_values=%s",
        config.function_spec.key,
        config.kind,
        config.method_key,
        config.interval,
        eps_values,
        l_values,
    )
    if not eps_values:
        raise ValueError("Список ε для запуска пуст.")
    if not l_values:
        raise ValueError("Список l для запуска пуст.")

    method_keys = resolve_method_keys(config.method_key)
    best_by_method: Dict[str, list[tuple[float, float, SearchResult]]] = {key: [] for key in method_keys}
    skipped_runs: list[SkippedRun] = []
    grid_runs_by_method: Dict[str, list[GridRunResult]] = {key: [] for key in method_keys}

    for eps in eps_values:
        for l_value in l_values:
            logger.debug("run_batch combination eps=%s l=%s", eps, l_value)
            for method_key in method_keys:
                if not is_valid_method_params(method_key, eps, l_value):
                    reason = skip_reason(method_key)
                    logger.info(
                        "run_batch method=%s skipped eps=%s l=%s reason=%s",
                        method_key,
                        eps,
                        l_value,
                        reason,
                    )
                    skipped_runs.append(SkippedRun(method_key=method_key, eps=eps, l=l_value, reason=reason))
                    continue
                try:
                    result = METHOD_SPECS[method_key].runner(
                        config.function_spec.func,
                        config.interval[0],
                        config.interval[1],
                        eps,
                        l_value,
                        config.kind,
                    )
                except (ValueError, ZeroDivisionError) as method_error:
                    logger.warning(
                        "run_batch method=%s failed eps=%s l=%s error=%s",
                        method_key,
                        eps,
                        l_value,
                        method_error,
                    )
                    continue

                best_by_method[method_key].append((eps, l_value, result))
                grid_runs_by_method[method_key].append(
                    GridRunResult(method_key=method_key, eps=eps, l=l_value, result=result)
                )
                logger.debug(
                    "run_batch method=%s success eps=%s l=%s x_opt=%.10f f_opt=%.10f evals=%d",
                    method_key,
                    eps,
                    l_value,
                    result.x_opt,
                    result.f_opt,
                    result.func_evals,
                )

    results_by_method: Dict[str, SearchResult] = {}
    for method_key in method_keys:
        runs = best_by_method[method_key]
        if not runs:
            continue
        best_run = min(runs, key=lambda item: item[2].func_evals)
        results_by_method[method_key] = best_run[2]

    if not results_by_method:
        raise ValueError("Ни один метод не смог выполнить расчёт с текущими параметрами.")

    available_method_keys = tuple(key for key in method_keys if key in results_by_method)
    is_single_mode = len(eps_values) == 1 and len(l_values) == 1
    report = RunReport(
        results_by_method=results_by_method,
        method_keys=available_method_keys,
        default_method_key=next(iter(results_by_method), None),
        requested_method_key=config.method_key,
        function_spec=config.function_spec,
        kind=config.kind,
        interval_raw=config.interval_raw,
        interval=config.interval,
        eps=eps_values[0] if is_single_mode else None,
        l=l_values[0] if is_single_mode else None,
        plot_range=build_plot_range(config.function_spec, config.interval_raw),
        reference_point=theoretical_optimum(config),
        mode="single" if is_single_mode else "grid",
        grid_runs_by_method={key: tuple(grid_runs_by_method[key]) for key in available_method_keys},
        skipped_runs=tuple(skipped_runs),
    )
    logger.info(
        "run_batch done result_methods=%s runs=%d skipped=%d mode=%s duration_ms=%.2f",
        available_method_keys,
        sum(len(grid_runs_by_method[key]) for key in available_method_keys),
        len(skipped_runs),
        report.mode,
        (time.perf_counter() - started) * 1000.0,
    )
    return report
