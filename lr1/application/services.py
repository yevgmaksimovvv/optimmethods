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

from lr1.application.analysis import (
    build_plot_range,
    is_valid_method_params,
    resolve_method_keys,
    skip_reason,
    theoretical_optimum,
)
from lr1.domain.functions import FUNCTION_TEMPLATE_SPECS, build_function_spec
from lr1.domain.models import GridRunResult, InputConfig, RunReport, SearchResult, SkippedRun
from lr1.domain.numerical import sanitize_interval, scaled_interval_shift
from lr1.domain.search import METHOD_SPECS
from lr1.infrastructure.logging import configure_logging
from lr1.infrastructure.settings import GRID_L_VALUES, SERIES_EPS_VALUES


configure_logging()
logger = logging.getLogger("lr1.app_service")


def _parse_float(raw_value: str, field_name: str) -> float:
    """Преобразует строку из интерфейса в число с поддержкой запятой.

    Пользователь в русской локали часто вводит `1,5` вместо `1.5`,
    поэтому функция нормализует разделитель и выбрасывает понятную ошибку
    с именем конкретного поля.
    """
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
    coefficient_raws: Dict[str, str],
) -> InputConfig:
    """Собирает и валидирует все входные параметры одного расчёта.

    На этом шаге происходит вся подготовка данных перед запуском алгоритмов:
    - проверяется существование выбранной функции и метода;
    - читаются числовые поля интерфейса;
    - строится `FunctionSpec` по коэффициентам;
    - интервал очищается от недопустимых точек разрыва.

    Возвращается полностью готовый `InputConfig`, который уже можно
    без дополнительных проверок передавать в `run_single` или `run_full_grid`.
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

    a = _parse_float(a_raw, "a")
    b = _parse_float(b_raw, "b")
    eps = _parse_float(eps_raw, "ε")
    l = _parse_float(l_raw, "l")

    if a >= b:
        raise ValueError("Должно быть a < b.")
    if eps <= 0 or l <= 0:
        raise ValueError("Параметры ε и l должны быть положительными.")

    template = FUNCTION_TEMPLATE_SPECS[function_key]
    coefficients = {
        item.key: _parse_float(coefficient_raws.get(item.key, str(item.default)), item.label)
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
        l=l,
    )
    logger.info("build_input_config success function=%s interval=%s", config.function_spec.key, config.interval)
    return config


def run_single(config: InputConfig) -> RunReport:
    """Выполняет одиночный расчёт для одного метода или сразу для всех.

    Если пользователь выбрал `all`, функция прогоняет все доступные методы,
    собирает успешные результаты и формирует единый отчёт для вкладок UI.
    Методы, которые не смогли выполниться на текущих параметрах, просто
    пропускаются, чтобы приложение не падало из-за одного невалидного случая.
    """
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

    if not results_by_method:
        raise ValueError("Ни один метод не смог выполнить расчёт с текущими параметрами.")

    available_method_keys = tuple(key for key in method_keys if key in results_by_method)
    report = RunReport(
        results_by_method=results_by_method,
        method_keys=available_method_keys,
        default_method_key=next(iter(results_by_method), None),
        requested_method_key=config.method_key,
        function_spec=config.function_spec,
        kind=config.kind,
        interval_raw=config.interval_raw,
        interval=config.interval,
        eps=config.eps,
        l=config.l,
        plot_range=build_plot_range(config.function_spec, config.interval_raw),
        reference_point=theoretical_optimum(config),
        mode="single",
    )
    logger.info(
        "run_single done results=%d duration_ms=%.2f",
        len(available_method_keys),
        (time.perf_counter() - started) * 1000.0,
    )
    return report


def run_full_grid(config: InputConfig) -> RunReport:
    """Запускает серию расчётов по фиксированной сетке `ε × l`.

    В этом режиме пользовательский ввод задаёт функцию, тип экстремума и
    рабочий интервал, а сами пары `ε` и `l` берутся из конфигурации серии.
    Для каждой допустимой комбинации параметров функция:
    - пытается выполнить расчёт;
    - накапливает успешные прогоны;
    - отдельно фиксирует пропущенные невалидные сочетания.

    В итоговый `RunReport` попадают лучшие результаты по каждому методу,
    а также полный набор прогонов для сравнительных таблиц и графиков.
    """
    started = time.perf_counter()
    logger.info(
        "run_full_grid start function=%s kind=%s method=%s interval=%s",
        config.function_spec.key,
        config.kind,
        config.method_key,
        config.interval,
    )
    method_keys = resolve_method_keys(config.method_key)
    best_by_method: Dict[str, list[tuple[float, float, SearchResult]]] = {key: [] for key in method_keys}
    skipped_runs: list[SkippedRun] = []
    grid_runs_by_method: Dict[str, list[GridRunResult]] = {key: [] for key in method_keys}

    for eps in SERIES_EPS_VALUES:
        for l in GRID_L_VALUES:
            logger.debug("run_full_grid combination eps=%s l=%s", eps, l)
            for method_key in method_keys:
                if not is_valid_method_params(method_key, eps, l):
                    reason = skip_reason(method_key)
                    logger.info(
                        "run_full_grid method=%s skipped eps=%s l=%s reason=%s",
                        method_key,
                        eps,
                        l,
                        reason,
                    )
                    skipped_runs.append(SkippedRun(method_key=method_key, eps=eps, l=l, reason=reason))
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
                    continue

                best_by_method[method_key].append((eps, l, result))
                grid_runs_by_method[method_key].append(GridRunResult(method_key=method_key, eps=eps, l=l, result=result))
                logger.debug(
                    "run_full_grid method=%s success eps=%s l=%s x_opt=%.10f f_opt=%.10f evals=%d",
                    method_key,
                    eps,
                    l,
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

    available_method_keys = tuple(key for key in method_keys if key in results_by_method)
    report = RunReport(
        results_by_method=results_by_method,
        method_keys=available_method_keys,
        default_method_key=next(iter(results_by_method), None),
        requested_method_key=config.method_key,
        function_spec=config.function_spec,
        kind=config.kind,
        interval_raw=config.interval_raw,
        interval=config.interval,
        eps=None,
        l=None,
        plot_range=build_plot_range(config.function_spec, config.interval_raw),
        reference_point=theoretical_optimum(config),
        mode="grid",
        grid_runs_by_method={key: tuple(grid_runs_by_method[key]) for key in available_method_keys},
        skipped_runs=tuple(skipped_runs),
    )
    logger.info(
        "run_full_grid done result_methods=%s duration_ms=%.2f",
        available_method_keys,
        (time.perf_counter() - started) * 1000.0,
    )
    return report
