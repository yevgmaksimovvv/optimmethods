"""Прикладной слой ЛР3: валидация ввода и запуск методов."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import cast

from optim_core.parsing import parse_localized_float

from lr3.domain.expression import ExpressionError, compile_objective
from lr3.domain.methods import conjugate_gradient_ascent, gradient_ascent
from lr3.domain.models import Goal, MethodConfig, OptimizationResult, Point2D

logger = logging.getLogger("lr3.service")

DEFAULT_GRADIENT_EXPRESSION = "x1**2 + x2**2 - x1*x2 + x1 - 2*x2"
DEFAULT_CONJUGATE_EXPRESSION = "-2 - x1 - 2*x2 - 0.1*x1**2 - 100*x2**2"


@dataclass(frozen=True)
class ServiceMetrics:
    """Базовые метрики вычислительного запуска."""

    trace_id: str
    latency_ms: float
    success: bool


def parse_int(value: str, field_name: str) -> int:
    """Безопасный парсинг целого параметра."""
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Неверное целое число в поле '{field_name}': {value}") from exc


def build_start_point(x1_raw: str, x2_raw: str) -> Point2D:
    """Создаёт стартовую точку из UI-строк."""
    return (
        parse_localized_float(x1_raw, "x1"),
        parse_localized_float(x2_raw, "x2"),
    )


def build_config(
    epsilon_raw: str,
    max_iterations_raw: str,
    initial_step_raw: str,
    timeout_raw: str,
    goal_raw: str,
    min_step_raw: str = "1e-8",
    gradient_step_raw: str = "1e-6",
) -> MethodConfig:
    """Валидирует и строит конфиг метода."""
    epsilon = parse_localized_float(epsilon_raw, "epsilon")
    max_iterations = parse_int(max_iterations_raw, "max_iterations")
    initial_step = parse_localized_float(initial_step_raw, "initial_step")
    timeout_seconds = parse_localized_float(timeout_raw, "timeout_seconds")
    goal = _parse_goal(goal_raw)
    min_step = parse_localized_float(min_step_raw, "min_step")
    gradient_step = parse_localized_float(gradient_step_raw, "gradient_step")

    if epsilon <= 0:
        raise ValueError("epsilon должен быть > 0")
    if max_iterations <= 0:
        raise ValueError("max_iterations должен быть > 0")
    if initial_step <= 0:
        raise ValueError("initial_step должен быть > 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds должен быть > 0")
    if min_step <= 0:
        raise ValueError("min_step должен быть > 0")
    if gradient_step <= 0:
        raise ValueError("gradient_step должен быть > 0")

    return MethodConfig(
        epsilon=epsilon,
        max_iterations=max_iterations,
        initial_step=initial_step,
        min_step=min_step,
        timeout_seconds=timeout_seconds,
        goal=goal,
        gradient_step=gradient_step,
    )


def run_gradient(expression: str, start_point: Point2D, config: MethodConfig) -> tuple[OptimizationResult, ServiceMetrics]:
    """Запускает градиентный метод с метриками."""
    return _run(method="gradient", expression=expression, start_point=start_point, config=config)


def run_conjugate(expression: str, start_point: Point2D, config: MethodConfig) -> tuple[OptimizationResult, ServiceMetrics]:
    """Запускает метод сопряжённых градиентов с метриками."""
    return _run(method="conjugate", expression=expression, start_point=start_point, config=config)


def _run(
    method: str,
    expression: str,
    start_point: Point2D,
    config: MethodConfig,
) -> tuple[OptimizationResult, ServiceMetrics]:
    trace_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    logger.info(
        "run start trace_id=%s method=%s expression=%s start_point=%s",
        trace_id,
        method,
        expression,
        start_point,
    )

    try:
        objective = compile_objective(expression)
    except ExpressionError as exc:
        logger.error("run invalid_expression trace_id=%s error=%s", trace_id, exc)
        raise ValueError(str(exc)) from exc

    if method == "gradient":
        result = gradient_ascent(objective, start_point, config)
    else:
        result = conjugate_gradient_ascent(objective, start_point, config)

    metrics = ServiceMetrics(
        trace_id=trace_id,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        success=result.success,
    )
    logger.info(
        "run done trace_id=%s method=%s success=%s iterations=%s latency_ms=%.2f stop_reason=%s",
        trace_id,
        method,
        result.success,
        result.iterations_count,
        metrics.latency_ms,
        result.stop_reason,
    )
    return result, metrics


def _parse_goal(value: str) -> Goal:
    goal = value.strip().lower()
    if goal not in {"min", "max"}:
        raise ValueError("goal должен быть min или max")
    return cast(Goal, goal)
