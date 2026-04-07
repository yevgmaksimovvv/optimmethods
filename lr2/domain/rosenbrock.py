"""N-мерный метод Розенброка с минимизацией по направлению."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

from lr2.domain.models import DiscreteSolverConfig, IterationStep, SolverConfig, SolverResult, Vector

logger = logging.getLogger("lr2.rosenbrock")


def _vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _vector_scale(vector: Vector, scalar: float) -> Vector:
    return tuple(item * scalar for item in vector)


def _vector_sub(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: Vector) -> float:
    return math.sqrt(sum(item * item for item in vector))


def _safe_normalize(vector: Vector, fallback: Vector, threshold: float) -> Vector:
    length = _norm(vector)
    if length <= threshold:
        return fallback
    return tuple(item / length for item in vector)


def _canonical_basis(dimension: int) -> tuple[Vector, ...]:
    return tuple(
        tuple(1.0 if i == j else 0.0 for i in range(dimension))
        for j in range(dimension)
    )


def _modified_gram_schmidt(
    vectors: tuple[Vector, ...],
    fallback_basis: tuple[Vector, ...],
    zero_tolerance: float,
) -> tuple[tuple[Vector, ...], bool]:
    """Строит ортонормированный базис устойчивым MGS + детерминированный fallback."""
    dimension = len(vectors)
    canonical_basis = _canonical_basis(dimension)
    orthonormal: list[Vector] = []
    used_primary_vectors = 0
    candidates: list[Vector] = [*vectors, *fallback_basis, *canonical_basis]

    for candidate_idx, candidate in enumerate(candidates):
        ortho_candidate = candidate
        for basis_vector in orthonormal:
            projection = _dot(ortho_candidate, basis_vector)
            ortho_candidate = _vector_sub(ortho_candidate, _vector_scale(basis_vector, projection))

        candidate_norm = _norm(ortho_candidate)
        if candidate_norm <= zero_tolerance:
            continue
        orthonormal.append(tuple(value / candidate_norm for value in ortho_candidate))
        if candidate_idx < dimension:
            used_primary_vectors += 1
        if len(orthonormal) == dimension:
            break

    if len(orthonormal) != dimension:
        raise RuntimeError("Не удалось построить ортонормированный базис направлений.")

    degenerate = used_primary_vectors < dimension
    return tuple(orthonormal), degenerate


def _build_rotated_directions(
    previous_directions: tuple[Vector, ...],
    lambdas: tuple[float, ...],
    zero_tolerance: float,
) -> tuple[tuple[Vector, ...], bool]:
    """Перестраивает базис направлений по схеме Розенброка + modified Gram-Schmidt."""
    q_vectors: list[Vector] = []
    dimension = len(previous_directions)
    for start_idx in range(dimension):
        cumulative = tuple(0.0 for _ in range(dimension))
        for direction, lambda_value in zip(
            previous_directions[start_idx:],
            lambdas[start_idx:],
            strict=True,
        ):
            cumulative = _vector_add(cumulative, _vector_scale(direction, lambda_value))
        q_vectors.append(cumulative)
    return _modified_gram_schmidt(
        vectors=tuple(q_vectors),
        fallback_basis=previous_directions,
        zero_tolerance=zero_tolerance,
    )


def _reversed_canonical_basis(dimension: int) -> tuple[Vector, ...]:
    """Возвращает канонический базис в обратном порядке, как в дискретной схеме."""
    return tuple(
        tuple(1.0 if i == dimension - 1 - j else 0.0 for i in range(dimension))
        for j in range(dimension)
    )


def _build_discrete_directions(
    previous_directions: tuple[Vector, ...],
    delta_x: Vector,
    zero_tolerance: float,
) -> tuple[tuple[Vector, ...], bool]:
    """Перестраивает направления для дискретного варианта по гамма-шмитц схеме."""
    lambda_array = tuple(_dot(delta_x, direction) for direction in previous_directions)
    a_vectors = tuple(
        _vector_add(direction, _vector_scale(delta_x, lambda_value))
        for direction, lambda_value in zip(previous_directions, lambda_array, strict=True)
    )
    return _modified_gram_schmidt(
        vectors=a_vectors,
        fallback_basis=previous_directions,
        zero_tolerance=zero_tolerance,
    )


def _golden_section_minimize(
    phi: Callable[[float], float],
    left: float,
    right: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Минимизация на отрезке методом золотого сечения."""
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = phi(x1)
    f2 = phi(x2)

    for _ in range(max_iterations):
        if abs(right - left) <= tolerance:
            break
        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - golden * (right - left)
            f1 = phi(x1)
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + golden * (right - left)
            f2 = phi(x2)

    return (left + right) / 2.0


def _line_search(phi: Callable[[float], float], config: SolverConfig) -> float:
    # Для плоского профиля выбираем нулевой шаг, чтобы избежать случайного дрейфа
    # на эквивалентных значениях функции.
    if config.line_search_min_lambda <= 0.0 <= config.line_search_max_lambda:
        probe_points = (
            config.line_search_min_lambda,
            0.0,
            config.line_search_max_lambda,
        )
        probe_values = [phi(point) for point in probe_points]
        spread = max(probe_values) - min(probe_values)
        scale = max(1.0, abs(probe_values[1]))
        if spread <= config.line_search_tolerance * scale:
            return 0.0

    return _golden_section_minimize(
        phi,
        left=config.line_search_min_lambda,
        right=config.line_search_max_lambda,
        tolerance=config.line_search_tolerance,
        max_iterations=config.line_search_max_iterations,
    )


def _stagnation_tolerance(current_value: float, next_value: float, config: SolverConfig) -> float:
    return config.stagnation_abs_tolerance + (
        config.stagnation_rel_tolerance * max(1.0, abs(current_value), abs(next_value))
    )


def _validate_start_point(start_point: Vector) -> None:
    if len(start_point) < 2:
        raise ValueError("Размерность start_point должна быть >= 2")


def rosenbrock_minimize(
    objective: Callable[[Vector], float],
    start_point: Vector,
    config: SolverConfig,
) -> SolverResult:
    """Ищет минимум произвольной функции `objective` N-мерным методом Розенброка."""
    if config.epsilon <= 0.0:
        raise ValueError("epsilon должен быть > 0")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations должен быть > 0")
    if config.line_search_min_lambda >= config.line_search_max_lambda:
        raise ValueError("line_search_min_lambda должен быть < line_search_max_lambda")
    if config.line_search_tolerance <= 0.0:
        raise ValueError("line_search_tolerance должен быть > 0")
    if config.line_search_max_iterations <= 0:
        raise ValueError("line_search_max_iterations должен быть > 0")
    if config.direction_zero_tolerance <= 0.0:
        raise ValueError("direction_zero_tolerance должен быть > 0")
    if config.stagnation_abs_tolerance < 0.0:
        raise ValueError("stagnation_abs_tolerance должен быть >= 0")
    if config.stagnation_rel_tolerance < 0.0:
        raise ValueError("stagnation_rel_tolerance должен быть >= 0")

    _validate_start_point(start_point)
    dimension = len(start_point)
    directions: tuple[Vector, ...] = _canonical_basis(dimension)
    x_k = start_point
    trajectory: list[Vector] = [x_k]
    rows: list[IterationStep] = []

    for k in range(1, config.max_iterations + 1):
        f_x_k = objective(x_k)
        y_j = x_k
        lambdas: list[float] = []

        for j, direction in enumerate(directions, start=1):
            f_y_j = objective(y_j)

            def phi(lam: float, base: Vector = y_j, search_dir: Vector = direction) -> float:
                point = _vector_add(base, _vector_scale(search_dir, lam))
                return objective(point)

            lambda_j = _line_search(phi, config)
            y_next = _vector_add(y_j, _vector_scale(direction, lambda_j))
            f_y_next = objective(y_next)
            lambdas.append(lambda_j)
            rows.append(
                IterationStep(
                    k=k,
                    x_k=x_k,
                    f_x_k=f_x_k,
                    j=j,
                    direction=direction,
                    y_j=y_j,
                    f_y_j=f_y_j,
                    lambda_j=lambda_j,
                    y_next=y_next,
                    f_y_next=f_y_next,
                )
            )
            y_j = y_next

        x_next = y_j
        trajectory.append(x_next)
        f_next = objective(x_next)
        step_norm = _norm(_vector_sub(x_next, x_k))
        if step_norm <= config.epsilon:
            return SolverResult(
                epsilon=config.epsilon,
                start_point=start_point,
                optimum_point=x_next,
                optimum_value=f_next,
                iterations_count=k,
                steps=tuple(rows),
                trajectory=tuple(trajectory),
                success=True,
                stop_reason="Достигнут критерий ||x_(k+1) - x_k|| <= epsilon",
            )

        if abs(f_next - f_x_k) <= _stagnation_tolerance(f_x_k, f_next, config):
            return SolverResult(
                epsilon=config.epsilon,
                start_point=start_point,
                optimum_point=x_next,
                optimum_value=f_next,
                iterations_count=k,
                steps=tuple(rows),
                trajectory=tuple(trajectory),
                success=True,
                stop_reason="Достигнут критерий стагнации по изменению f(x)",
            )

        directions, was_degenerate = _build_rotated_directions(
            previous_directions=directions,
            lambdas=tuple(lambdas),
            zero_tolerance=config.direction_zero_tolerance,
        )
        if was_degenerate:
            logger.info(
                "direction_basis_degenerate iteration=%d step_norm=%.3e f_delta=%.3e",
                k,
                step_norm,
                abs(f_next - f_x_k),
            )
        x_k = x_next

    final_value = objective(x_k)
    return SolverResult(
        epsilon=config.epsilon,
        start_point=start_point,
        optimum_point=x_k,
        optimum_value=final_value,
        iterations_count=config.max_iterations,
        steps=tuple(rows),
        trajectory=tuple(trajectory),
        success=False,
        stop_reason="Достигнут лимит итераций",
    )


def discrete_rosenbrock_minimize(
    objective: Callable[[Vector], float],
    start_point: Vector,
    config: DiscreteSolverConfig,
) -> SolverResult:
    """Ищет минимум дискретным вариантом метода Розенброка."""
    if config.epsilon <= 0.0:
        raise ValueError("epsilon должен быть > 0")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations должен быть > 0")
    if config.delta_step <= 0.0:
        raise ValueError("delta_step должен быть > 0")
    if config.alpha <= 1.0:
        raise ValueError("alpha должен быть > 1")
    if not (-1.0 < config.beta < 0.0):
        raise ValueError("beta должен быть в диапазоне (-1, 0)")
    if config.direction_zero_tolerance <= 0.0:
        raise ValueError("direction_zero_tolerance должен быть > 0")

    _validate_start_point(start_point)
    dimension = len(start_point)
    directions: tuple[Vector, ...] = _reversed_canonical_basis(dimension)
    delta_values = [config.delta_step for _ in range(dimension)]
    x_k = start_point
    trajectory: list[Vector] = [x_k]
    rows: list[IterationStep] = []
    k = 1

    while k <= config.max_iterations:
        f_x_k = objective(x_k)
        y_j = x_k
        moved = False

        for j, direction in enumerate(directions, start=1):
            f_y_j = objective(y_j)
            delta_j = delta_values[j - 1]
            y_next = _vector_add(y_j, _vector_scale(direction, delta_j))
            f_y_next = objective(y_next)
            accepted = f_y_next < f_y_j

            rows.append(
                IterationStep(
                    k=k,
                    x_k=x_k,
                    f_x_k=f_x_k,
                    j=j,
                    direction=direction,
                    y_j=y_j,
                    f_y_j=f_y_j,
                    lambda_j=delta_j,
                    y_next=y_next,
                    f_y_next=f_y_next,
                )
            )

            if accepted:
                y_j = y_next
                delta_values[j - 1] = config.alpha * delta_j
                moved = True
            else:
                delta_values[j - 1] = config.beta * delta_j

        if moved:
            delta_x = _vector_sub(y_j, x_k)
            if _norm(delta_x) <= config.epsilon:
                return SolverResult(
                    epsilon=config.epsilon,
                    start_point=start_point,
                    optimum_point=y_j,
                    optimum_value=objective(y_j),
                    iterations_count=k,
                    steps=tuple(rows),
                    trajectory=tuple([*trajectory, y_j]),
                    success=True,
                    stop_reason="Достигнут критерий ||x_(k+1) - x_k|| <= epsilon",
                )

            directions, was_degenerate = _build_discrete_directions(
                previous_directions=directions,
                delta_x=delta_x,
                zero_tolerance=config.direction_zero_tolerance,
            )
            if was_degenerate:
                logger.info(
                    "discrete_direction_basis_degenerate iteration=%d step_norm=%.3e",
                    k,
                    _norm(delta_x),
                )

            x_k = y_j
            trajectory.append(x_k)
            k += 1
            continue

        if abs(delta_values[-1]) <= config.epsilon:
            return SolverResult(
                epsilon=config.epsilon,
                start_point=start_point,
                optimum_point=x_k,
                optimum_value=f_x_k,
                iterations_count=k,
                steps=tuple(rows),
                trajectory=tuple(trajectory),
                success=True,
                stop_reason="Достигнут критерий |Δ_j| <= epsilon",
            )

    return SolverResult(
        epsilon=config.epsilon,
        start_point=start_point,
        optimum_point=x_k,
        optimum_value=objective(x_k),
        iterations_count=config.max_iterations,
        steps=tuple(rows),
        trajectory=tuple(trajectory),
        success=False,
        stop_reason="Достигнут лимит итераций",
    )
