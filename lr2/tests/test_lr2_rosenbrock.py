"""Контрактные тесты ЛР2.

Эталоны минимумов взяты из стандартных benchmark-функций:
- Rosenbrock/Booth/Matyas/Himmelblau:
  https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import math
from dataclasses import dataclass

import pytest
from lr2.application.services import DEFAULT_SOLVER_CONFIG, VARIANT_PRESETS, build_polynomial
from lr2.domain.models import Polynomial2D, SolverConfig, Vector
from lr2.domain.polynomial import evaluate_polynomial
from lr2.domain.rosenbrock import rosenbrock_minimize


def _config(epsilon: float) -> SolverConfig:
    return SolverConfig(
        epsilon=epsilon,
        max_iterations=int(DEFAULT_SOLVER_CONFIG["max_iterations"]),
        line_search_min_lambda=float(DEFAULT_SOLVER_CONFIG["line_search_min_lambda"]),
        line_search_max_lambda=float(DEFAULT_SOLVER_CONFIG["line_search_max_lambda"]),
        line_search_tolerance=float(DEFAULT_SOLVER_CONFIG["line_search_tolerance"]),
        line_search_max_iterations=int(DEFAULT_SOLVER_CONFIG["line_search_max_iterations"]),
        direction_zero_tolerance=float(DEFAULT_SOLVER_CONFIG["direction_zero_tolerance"]),
        stagnation_abs_tolerance=float(DEFAULT_SOLVER_CONFIG["stagnation_abs_tolerance"]),
        stagnation_rel_tolerance=float(DEFAULT_SOLVER_CONFIG["stagnation_rel_tolerance"]),
    )


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    polynomial: Polynomial2D
    start_point: Vector
    minima: tuple[Vector, ...]
    expected_value: float
    tolerance_point: float
    tolerance_value: float
    iteration_budget: dict[float, int]


def _distance(point_a: Vector, point_b: Vector) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b, strict=True)))


BENCHMARK_CASES = (
    BenchmarkCase(
        key="variant_f1",
        polynomial=build_polynomial("F1", VARIANT_PRESETS["variant_f1"]),
        start_point=(0.0, 1.0),
        minima=((2.0 / 9.0, 4.0 / 27.0),),
        expected_value=0.0,
        tolerance_point=2e-2,
        tolerance_value=1e-4,
        iteration_budget={0.1: 6, 0.01: 9, 0.001: 9},
    ),
    BenchmarkCase(
        key="variant_f2",
        polynomial=build_polynomial("F2", VARIANT_PRESETS["variant_f2"]),
        start_point=(0.0, 0.0),
        minima=((5.0, 4.0),),
        expected_value=-481.0,
        tolerance_point=5e-2,
        tolerance_value=2e-1,
        iteration_budget={0.1: 14, 0.01: 14, 0.001: 14},
    ),
    BenchmarkCase(
        key="booth",
        polynomial=build_polynomial(
            "Booth",
            (
                (74.0, -38.0, 5.0),
                (-34.0, 8.0, 0.0),
                (5.0, 0.0, 0.0),
            ),
        ),
        start_point=(8.0, 9.0),
        minima=((1.0, 3.0),),
        expected_value=0.0,
        tolerance_point=2e-2,
        tolerance_value=1e-4,
        iteration_budget={0.1: 17, 0.01: 19, 0.001: 21},
    ),
    BenchmarkCase(
        key="matyas",
        polynomial=build_polynomial(
            "Matyas",
            (
                (0.0, 0.0, 0.26),
                (0.0, -0.48, 0.0),
                (0.26, 0.0, 0.0),
            ),
        ),
        start_point=(7.0, -9.0),
        minima=((0.0, 0.0),),
        expected_value=0.0,
        tolerance_point=2e-2,
        tolerance_value=1e-4,
        iteration_budget={0.1: 24, 0.01: 24, 0.001: 25},
    ),
    BenchmarkCase(
        key="rosenbrock_classic",
        polynomial=build_polynomial(
            "Rosenbrock",
            (
                (1.0, 0.0, 100.0),
                (-2.0, 0.0, 0.0),
                (1.0, -200.0, 0.0),
                (0.0, 0.0, 0.0),
                (100.0, 0.0, 0.0),
            ),
        ),
        start_point=(-1.2, 1.0),
        minima=((1.0, 1.0),),
        expected_value=0.0,
        tolerance_point=8e-2,
        tolerance_value=5e-2,
        iteration_budget={0.1: 3, 0.01: 25, 0.001: 26},
    ),
    BenchmarkCase(
        key="himmelblau",
        polynomial=build_polynomial(
            "Himmelblau",
            (
                (170.0, -22.0, -13.0, 0.0, 1.0),
                (-14.0, 0.0, 2.0, 0.0, 0.0),
                (-21.0, 2.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0, 0.0),
            ),
        ),
        start_point=(0.0, 0.0),
        minima=((3.0, 2.0), (-2.805118, 3.131312), (-3.77931, -3.283186), (3.584428, -1.848126)),
        expected_value=0.0,
        tolerance_point=9e-2,
        tolerance_value=2e-3,
        iteration_budget={0.1: 9, 0.01: 9, 0.001: 11},
    ),
)


@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=[case.key for case in BENCHMARK_CASES])
def test_rosenbrock_converges_to_known_minimum(case: BenchmarkCase) -> None:
    def objective(vector: Vector) -> float:
        assert len(vector) == 2
        return evaluate_polynomial(case.polynomial, vector[0], vector[1])

    result = rosenbrock_minimize(
        objective=objective,
        start_point=case.start_point,
        config=_config(1e-3),
    )

    assert result.success
    assert min(_distance(result.optimum_point, minimum) for minimum in case.minima) <= case.tolerance_point
    assert abs(result.optimum_value - case.expected_value) <= case.tolerance_value


@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=[case.key for case in BENCHMARK_CASES])
@pytest.mark.parametrize("epsilon", (0.1, 0.01, 0.001))
def test_rosenbrock_iteration_budget_by_epsilon(case: BenchmarkCase, epsilon: float) -> None:
    def objective(vector: Vector) -> float:
        assert len(vector) == 2
        return evaluate_polynomial(case.polynomial, vector[0], vector[1])

    result = rosenbrock_minimize(
        objective=objective,
        start_point=case.start_point,
        config=_config(epsilon),
    )

    assert result.success
    assert result.iterations_count <= case.iteration_budget[epsilon]


def test_rosenbrock_zero_polynomial_stops_immediately() -> None:
    polynomial = build_polynomial("Zero", tuple(tuple(0.0 for _ in range(5)) for _ in range(5)))
    start_point = (0.0, 1.0)

    def objective(vector: Vector) -> float:
        return evaluate_polynomial(polynomial, vector[0], vector[1])

    result = rosenbrock_minimize(
        objective=objective,
        start_point=start_point,
        config=_config(0.1),
    )

    assert result.success
    assert result.iterations_count == 1
    assert result.optimum_point == pytest.approx(start_point)
    assert result.optimum_value == pytest.approx(0.0)
    assert len(result.steps) == 2
    assert all(step.lambda_j == pytest.approx(0.0) for step in result.steps)
    assert all(step.y_j == pytest.approx(start_point) for step in result.steps)
    assert all(step.y_next == pytest.approx(start_point) for step in result.steps)


def test_rosenbrock_nd_quadratic_converges() -> None:
    def objective(vector: Vector) -> float:
        return sum((value - 1.0) ** 2 for value in vector)

    result = rosenbrock_minimize(
        objective=objective,
        start_point=(4.0, -3.0, 7.0),
        config=_config(1e-3),
    )

    assert result.success
    assert result.optimum_point == pytest.approx((1.0, 1.0, 1.0), abs=5e-2)
    assert result.optimum_value == pytest.approx(0.0, abs=1e-3)


def test_rosenbrock_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="Размерность start_point должна быть >= 2"):
        rosenbrock_minimize(
            objective=lambda vector: sum(value * value for value in vector),
            start_point=(1.0,),
            config=_config(1e-3),
        )


def test_rosenbrock_uses_hard_lambda_bounds_without_expansion() -> None:
    polynomial = build_polynomial("F2", VARIANT_PRESETS["variant_f2"])

    def objective(vector: Vector) -> float:
        return evaluate_polynomial(polynomial, vector[0], vector[1])

    result = rosenbrock_minimize(
        objective=objective,
        start_point=(-8.0, 10.0),
        config=_config(1e-3),
    )

    assert result.steps
    min_lambda = float(DEFAULT_SOLVER_CONFIG["line_search_min_lambda"])
    max_lambda = float(DEFAULT_SOLVER_CONFIG["line_search_max_lambda"])
    assert all(min_lambda <= step.lambda_j <= max_lambda for step in result.steps)
    assert result.steps[0].lambda_j == pytest.approx(max_lambda, abs=1e-3)
