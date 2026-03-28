"""Прикладной слой ЛР2: валидация ввода и пакетный запуск."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Final

from lr2.domain.models import BatchResult, Polynomial2D, SolverConfig, SolverResult, normalize_coefficients
from lr2.domain.polynomial import evaluate_polynomial, format_polynomial
from lr2.domain.rosenbrock import rosenbrock_minimize

logger = logging.getLogger("lr2.service")


@dataclass(frozen=True)
class ServiceMetrics:
    """Базовые метрики выполнения серии запусков."""

    trace_id: str
    run_count: int
    success_count: int
    error_count: int
    latency_ms: float


DEFAULT_SOLVER_CONFIG: dict[str, int | float] = {
    "max_iterations": 200,
    "line_search_initial_step": 0.5,
    "line_search_growth": 1.8,
    "line_search_max_expand": 22,
    "line_search_samples": 17,
    "line_search_tolerance": 1e-5,
    "line_search_max_iterations": 120,
    "direction_zero_tolerance": 1e-12,
    "stagnation_abs_tolerance": 1e-10,
    "stagnation_rel_tolerance": 1e-10,
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


def parse_float(value: str, field_name: str) -> float:
    """Парсит float с поддержкой запятой."""
    try:
        return float(value.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Неверное число в поле '{field_name}': {value}") from exc


def parse_epsilons(raw: str) -> tuple[float, ...]:
    """Парсит список epsilon через запятую."""
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise ValueError("Список epsilon пуст.")
    values = tuple(parse_float(item, "epsilon") for item in parts)
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
        points.append((parse_float(parts[0], "x1"), parse_float(parts[1], "x2")))

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
    errors = 0
    for epsilon in epsilons:
        config_payload = dict(DEFAULT_SOLVER_CONFIG)
        if max_iterations is not None:
            config_payload["max_iterations"] = max_iterations
        config = SolverConfig(
            epsilon=epsilon,
            max_iterations=int(config_payload["max_iterations"]),
            line_search_initial_step=float(config_payload["line_search_initial_step"]),
            line_search_growth=float(config_payload["line_search_growth"]),
            line_search_max_expand=int(config_payload["line_search_max_expand"]),
            line_search_samples=int(config_payload["line_search_samples"]),
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
            except Exception:
                errors += 1
                logger.exception(
                    "run_batch failed trace_id=%s epsilon=%s start_point=%s",
                    trace_id,
                    epsilon,
                    start_point,
                )
                continue
            runs.append(run)
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
        run_count=len(runs),
        success_count=sum(1 for item in runs if item.success),
        error_count=errors,
        latency_ms=latency_ms,
    )
    logger.info(
        "run_batch done trace_id=%s run_count=%d success_count=%d error_count=%d latency_ms=%.2f",
        metrics.trace_id,
        metrics.run_count,
        metrics.success_count,
        metrics.error_count,
        metrics.latency_ms,
    )
    return BatchResult(polynomial=polynomial, runs=tuple(runs)), metrics


def polynomial_value(polynomial: Polynomial2D, vector: tuple[float, ...]) -> float:
    """Вычисляет значение 2D-полинома в точке размерности 2."""
    if len(vector) != POLYNOMIAL_DIMENSION:
        raise ValueError(
            f"Ожидалась точка размерности {POLYNOMIAL_DIMENSION}, получено {len(vector)}"
        )
    return evaluate_polynomial(polynomial, vector[0], vector[1])
