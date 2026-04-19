"""Прикладной слой ЛР2: валидация ввода и пакетный запуск."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Final

from optim_core.parsing import parse_localized_float, parse_localized_float_sequence

from lr2.domain.models import (
    BatchResult,
    BatchItemResult,
    BatchSummary,
    BATCH_STATUS_DOMAIN_REFUSAL,
    BATCH_STATUS_SUCCESS,
    BATCH_STATUS_UNEXPECTED_ERROR,
    DiscreteSolverConfig,
    Polynomial2D,
    SolverConfig,
    SolverResult,
    normalize_coefficients,
)
from lr2.domain.polynomial import evaluate_polynomial, format_polynomial
from lr2.domain.rosenbrock import discrete_rosenbrock_minimize, rosenbrock_minimize

logger = logging.getLogger("lr2.service")


@dataclass(frozen=True)
class ServiceMetrics:
    """Базовые метрики выполнения серии запусков."""

    trace_id: str
    total_count: int
    success_count: int
    domain_refusal_count: int
    unexpected_error_count: int
    latency_ms: float

    @property
    def run_count(self) -> int:
        return self.success_count

    @property
    def failure_count(self) -> int:
        return self.domain_refusal_count + self.unexpected_error_count

    @property
    def error_count(self) -> int:
        return self.failure_count


DEFAULT_SOLVER_CONFIG: dict[str, int | float] = {
    "max_iterations": 200,
    "line_search_min_lambda": -0.5,
    "line_search_max_lambda": 0.5,
    "line_search_tolerance": 1e-5,
    "line_search_max_iterations": 120,
    "direction_zero_tolerance": 1e-12,
    "stagnation_abs_tolerance": 1e-10,
    "stagnation_rel_tolerance": 1e-10,
}

DEFAULT_DISCRETE_SOLVER_CONFIG: dict[str, int | float] = {
    "max_iterations": 200,
    "delta_step": 0.2,
    "alpha": 1.4,
    "beta": -0.2,
    "direction_zero_tolerance": 1e-12,
}

POLYNOMIAL_DIMENSION: Final[int] = 2


VARIANT_PRESETS: dict[str, tuple[tuple[float, ...], ...]] = {
    "variant_f1": (
        (0.0, 0.0, 10.0),
        (0.0, -12.0, 0.0),
        (4.0, -6.0, 0.0),
        (0.0, 0.0, 0.0),
        (9.0, 0.0, 0.0),
    ),
    "variant_f2": (
        (0.0, -128.0, 16.0),
        (-90.0, 0.0, 0.0),
        (9.0, 0.0, 0.0),
    ),
}


def parse_epsilons(raw: str) -> tuple[float, ...]:
    """Парсит список epsilon через запятую."""
    values = parse_localized_float_sequence(raw, "epsilon")
    if any(item <= 0 for item in values):
        raise ValueError("Все epsilon должны быть > 0.")
    return values


def parse_points(raw: str) -> tuple[tuple[float, float], ...]:
    """Парсит список стартовых точек формата: x1;x2 | x1;x2."""
    chunks = [item.strip() for item in raw.replace("\n", "|").split("|") if item.strip()]
    if not chunks:
        raise ValueError("Список стартовых точек пуст.")

    points: list[tuple[float, float]] = []
    for chunk in chunks:
        parts = [part.strip() for part in chunk.split(";")]
        if len(parts) != 2:
            raise ValueError(
                "Неверный формат стартовой точки. Используй 'x1;x2', "
                "а точки разделяй символом '|'."
            )
        points.append((parse_localized_float(parts[0], "x1"), parse_localized_float(parts[1], "x2")))

    return tuple(points)


def build_polynomial(title: str, matrix: tuple[tuple[float, ...], ...]) -> Polynomial2D:
    """Создает полином из матрицы коэффициентов."""
    normalized = normalize_coefficients(matrix)
    return Polynomial2D(coefficients=normalized, title=title)


def run_batch(
    polynomial: Polynomial2D,
    epsilons: tuple[float, ...],
    start_points: tuple[tuple[float, float], ...],
    max_iterations: int | None = None,
) -> tuple[BatchResult, ServiceMetrics]:
    """Запускает серию расчетов для всех комбинаций epsilon и стартовых точек."""
    trace_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    logger.info(
        "run_batch start trace_id=%s title=%s formula=%s epsilons=%s start_points=%s",
        trace_id,
        polynomial.title,
        format_polynomial(polynomial),
        epsilons,
        start_points,
    )

    runs: list[SolverResult] = []
    items: list[BatchItemResult] = []
    domain_refusal_count = 0
    unexpected_error_count = 0
    for epsilon in epsilons:
        config_payload = dict(DEFAULT_SOLVER_CONFIG)
        if max_iterations is not None:
            config_payload["max_iterations"] = max_iterations
        config = SolverConfig(
            epsilon=epsilon,
            max_iterations=int(config_payload["max_iterations"]),
            line_search_min_lambda=float(config_payload["line_search_min_lambda"]),
            line_search_max_lambda=float(config_payload["line_search_max_lambda"]),
            line_search_tolerance=float(config_payload["line_search_tolerance"]),
            line_search_max_iterations=int(config_payload["line_search_max_iterations"]),
            direction_zero_tolerance=float(config_payload["direction_zero_tolerance"]),
            stagnation_abs_tolerance=float(config_payload["stagnation_abs_tolerance"]),
            stagnation_rel_tolerance=float(config_payload["stagnation_rel_tolerance"]),
        )
        for start_point in start_points:
            if len(start_point) != POLYNOMIAL_DIMENSION:
                raise ValueError(
                    f"Ожидалась стартовая точка размерности {POLYNOMIAL_DIMENSION}, "
                    f"получено {len(start_point)}"
                )

            def objective(vector: tuple[float, ...]) -> float:
                if len(vector) != POLYNOMIAL_DIMENSION:
                    raise ValueError(
                        f"Ожидался вектор размерности {POLYNOMIAL_DIMENSION}, получено {len(vector)}"
                    )
                return polynomial_value(polynomial, vector)

            try:
                run = rosenbrock_minimize(objective=objective, start_point=start_point, config=config)
            except ValueError as exc:
                domain_refusal_count += 1
                items.append(
                    BatchItemResult(
                        epsilon=epsilon,
                        start_point=start_point,
                        status=BATCH_STATUS_DOMAIN_REFUSAL,
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                )
                logger.info(
                    "run_batch domain refusal trace_id=%s epsilon=%s start_point=%s reason=%s",
                    trace_id,
                    epsilon,
                    start_point,
                    exc,
                )
                continue
            except Exception as exc:
                unexpected_error_count += 1
                items.append(
                    BatchItemResult(
                        epsilon=epsilon,
                        start_point=start_point,
                        status=BATCH_STATUS_UNEXPECTED_ERROR,
                        message=str(exc) if exc is not None else "Неизвестная ошибка",
                        exception_type=type(exc).__name__ if exc is not None else "Exception",
                    )
                )
                logger.exception(
                    "run_batch unexpected error trace_id=%s epsilon=%s start_point=%s",
                    trace_id,
                    epsilon,
                    start_point,
                )
                continue
            runs.append(run)
            items.append(
                BatchItemResult(
                    epsilon=epsilon,
                    start_point=start_point,
                    status=BATCH_STATUS_SUCCESS,
                    run=run,
                )
            )
            logger.info(
                "run_batch item trace_id=%s epsilon=%s start=%s success=%s iterations=%s optimum=%s value=%.8f",
                trace_id,
                epsilon,
                start_point,
                run.success,
                run.iterations_count,
                run.optimum_point,
                run.optimum_value,
            )

    latency_ms = (time.perf_counter() - started) * 1000.0
    metrics = ServiceMetrics(
        trace_id=trace_id,
        total_count=len(items),
        success_count=sum(1 for item in runs if item.success),
        domain_refusal_count=domain_refusal_count,
        unexpected_error_count=unexpected_error_count,
        latency_ms=latency_ms,
    )
    logger.info(
        "run_batch done trace_id=%s total_count=%d success_count=%d domain_refusal_count=%d unexpected_error_count=%d latency_ms=%.2f",
        metrics.trace_id,
        metrics.total_count,
        metrics.success_count,
        metrics.domain_refusal_count,
        metrics.unexpected_error_count,
        metrics.latency_ms,
    )
    return BatchResult(
        polynomial=polynomial,
        runs=tuple(runs),
        items=tuple(items),
        summary=BatchSummary(
            total_count=len(items),
            success_count=len(runs),
            domain_refusal_count=domain_refusal_count,
            unexpected_error_count=unexpected_error_count,
        ),
    ), metrics


def run_discrete_batch(
    polynomial: Polynomial2D,
    epsilons: tuple[float, ...],
    start_points: tuple[tuple[float, float], ...],
    max_iterations: int | None = None,
    delta_step: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> tuple[BatchResult, ServiceMetrics]:
    """Запускает дискретный вариант метода для всех комбинаций параметров."""
    trace_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    logger.info(
        "run_discrete_batch start trace_id=%s title=%s formula=%s epsilons=%s start_points=%s",
        trace_id,
        polynomial.title,
        format_polynomial(polynomial),
        epsilons,
        start_points,
    )

    runs: list[SolverResult] = []
    items: list[BatchItemResult] = []
    domain_refusal_count = 0
    unexpected_error_count = 0
    for epsilon in epsilons:
        config_payload = dict(DEFAULT_DISCRETE_SOLVER_CONFIG)
        if max_iterations is not None:
            config_payload["max_iterations"] = max_iterations
        if delta_step is not None:
            config_payload["delta_step"] = delta_step
        if alpha is not None:
            config_payload["alpha"] = alpha
        if beta is not None:
            config_payload["beta"] = beta
        config = DiscreteSolverConfig(
            epsilon=epsilon,
            max_iterations=int(config_payload["max_iterations"]),
            delta_step=float(config_payload["delta_step"]),
            alpha=float(config_payload["alpha"]),
            beta=float(config_payload["beta"]),
            direction_zero_tolerance=float(config_payload["direction_zero_tolerance"]),
        )
        for start_point in start_points:
            if len(start_point) != POLYNOMIAL_DIMENSION:
                raise ValueError(
                    f"Ожидалась стартовая точка размерности {POLYNOMIAL_DIMENSION}, "
                    f"получено {len(start_point)}"
                )

            def objective(vector: tuple[float, ...]) -> float:
                if len(vector) != POLYNOMIAL_DIMENSION:
                    raise ValueError(
                        f"Ожидался вектор размерности {POLYNOMIAL_DIMENSION}, получено {len(vector)}"
                    )
                return polynomial_value(polynomial, vector)

            try:
                run = discrete_rosenbrock_minimize(
                    objective=objective,
                    start_point=start_point,
                    config=config,
                )
            except ValueError as exc:
                domain_refusal_count += 1
                items.append(
                    BatchItemResult(
                        epsilon=epsilon,
                        start_point=start_point,
                        status=BATCH_STATUS_DOMAIN_REFUSAL,
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                )
                logger.info(
                    "run_discrete_batch domain refusal trace_id=%s epsilon=%s start_point=%s reason=%s",
                    trace_id,
                    epsilon,
                    start_point,
                    exc,
                )
                continue
            except Exception as exc:
                unexpected_error_count += 1
                items.append(
                    BatchItemResult(
                        epsilon=epsilon,
                        start_point=start_point,
                        status=BATCH_STATUS_UNEXPECTED_ERROR,
                        message=str(exc) if exc is not None else "Неизвестная ошибка",
                        exception_type=type(exc).__name__ if exc is not None else "Exception",
                    )
                )
                logger.exception(
                    "run_discrete_batch unexpected error trace_id=%s epsilon=%s start_point=%s",
                    trace_id,
                    epsilon,
                    start_point,
                )
                continue
            runs.append(run)
            items.append(
                BatchItemResult(
                    epsilon=epsilon,
                    start_point=start_point,
                    status=BATCH_STATUS_SUCCESS,
                    run=run,
                )
            )
            logger.info(
                (
                    "run_discrete_batch item trace_id=%s epsilon=%s start=%s "
                    "success=%s iterations=%s optimum=%s value=%.8f"
                ),
                trace_id,
                epsilon,
                start_point,
                run.success,
                run.iterations_count,
                run.optimum_point,
                run.optimum_value,
            )

    latency_ms = (time.perf_counter() - started) * 1000.0
    metrics = ServiceMetrics(
        trace_id=trace_id,
        total_count=len(items),
        success_count=sum(1 for item in runs if item.success),
        domain_refusal_count=domain_refusal_count,
        unexpected_error_count=unexpected_error_count,
        latency_ms=latency_ms,
    )
    logger.info(
        "run_discrete_batch done trace_id=%s total_count=%d success_count=%d domain_refusal_count=%d unexpected_error_count=%d latency_ms=%.2f",
        metrics.trace_id,
        metrics.total_count,
        metrics.success_count,
        metrics.domain_refusal_count,
        metrics.unexpected_error_count,
        metrics.latency_ms,
    )
    return BatchResult(
        polynomial=polynomial,
        runs=tuple(runs),
        items=tuple(items),
        summary=BatchSummary(
            total_count=len(items),
            success_count=len(runs),
            domain_refusal_count=domain_refusal_count,
            unexpected_error_count=unexpected_error_count,
        ),
    ), metrics


def polynomial_value(polynomial: Polynomial2D, vector: tuple[float, ...]) -> float:
    """Вычисляет значение 2D-полинома в точке размерности 2."""
    if len(vector) != POLYNOMIAL_DIMENSION:
        raise ValueError(
            f"Ожидалась точка размерности {POLYNOMIAL_DIMENSION}, получено {len(vector)}"
        )
    return evaluate_polynomial(polynomial, vector[0], vector[1])
