"""Случайные, но аналитически проверяемые квадратичные тесты ЛР3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Literal

import pytest

from lr3.application.services import build_config, run_conjugate, run_gradient
from lr3.domain.expression import analyze_local_extremum, build_gradient_formula, build_hessian_formula, compile_objective
from lr3.domain.models import MethodConfig, Point2D

Kind = Literal["max", "min", "saddle"]


@dataclass(frozen=True)
class QuadraticCase:
    name: str
    kind: Kind
    expression: str
    hessian: tuple[tuple[float, float], tuple[float, float]]
    stationary_point: Point2D
    start_point: Point2D


def _config() -> MethodConfig:
    return build_config(
        epsilon_raw="1e-6",
        max_iterations_raw="450",
        initial_step_raw="0.1",
        timeout_raw="3.0",
        gradient_step_raw="1e-6",
        max_step_expansions_raw="18",
    )


def _generate_cases(kind: Kind, *, seed: int, count: int) -> list[QuadraticCase]:
    rng = Random(seed)
    cases: list[QuadraticCase] = []
    for index in range(count):
        hessian = _make_hessian(rng, kind)
        stationary_point = (rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5))
        start_point = _make_start_point(rng, stationary_point)
        expression = _build_expression(rng, hessian, stationary_point)
        cases.append(
            QuadraticCase(
                name=f"{kind}_{index + 1}",
                kind=kind,
                expression=expression,
                hessian=hessian,
                stationary_point=stationary_point,
                start_point=start_point,
            )
        )
    return cases


def _make_hessian(rng: Random, kind: Kind) -> tuple[tuple[float, float], tuple[float, float]]:
    if kind == "saddle":
        h11 = rng.uniform(0.9, 3.6)
        h22 = -rng.uniform(0.9, 3.6)
        h12 = rng.uniform(-2.0, 2.0)
        return ((h11, h12), (h12, h22))

    while True:
        h11 = rng.uniform(0.9, 3.6)
        h22 = rng.uniform(0.9, 3.6)
        if kind == "max":
            h11 = -h11
            h22 = -h22
        limit = 0.45 * math.sqrt(abs(h11 * h22))
        h12 = rng.uniform(-limit, limit)
        determinant = h11 * h22 - h12 * h12
        if determinant > 0.5:
            return ((h11, h12), (h12, h22))


def _make_start_point(rng: Random, stationary_point: Point2D) -> Point2D:
    for _ in range(100):
        candidate = (
            stationary_point[0] + rng.uniform(-2.5, 2.5),
            stationary_point[1] + rng.uniform(-2.5, 2.5),
        )
        if math.hypot(candidate[0] - stationary_point[0], candidate[1] - stationary_point[1]) >= 0.75:
            return candidate
    raise AssertionError("Не удалось сгенерировать стартовую точку с достаточным отступом")


def _build_expression(
    rng: Random,
    hessian: tuple[tuple[float, float], tuple[float, float]],
    stationary_point: Point2D,
) -> str:
    h11, h12 = hessian[0]
    _, h22 = hessian[1]
    x1_star, x2_star = stationary_point
    d1 = -(h11 * x1_star + h12 * x2_star)
    d2 = -(h12 * x1_star + h22 * x2_star)
    f = rng.uniform(-3.0, 3.0)
    a = h11 / 2.0
    c = h22 / 2.0
    return f"{a!r}*x1**2 + {h12!r}*x1*x2 + {c!r}*x2**2 + {d1!r}*x1 + {d2!r}*x2 + {f!r}"


def _expected_gradient(
    hessian: tuple[tuple[float, float], tuple[float, float]],
    stationary_point: Point2D,
    point: Point2D,
) -> Point2D:
    h11, h12 = hessian[0]
    _, h22 = hessian[1]
    x1_star, x2_star = stationary_point
    d1 = -(h11 * x1_star + h12 * x2_star)
    d2 = -(h12 * x1_star + h22 * x2_star)
    return (
        h11 * point[0] + h12 * point[1] + d1,
        h12 * point[0] + h22 * point[1] + d2,
    )


def _assert_analytical_block(case: QuadraticCase, *, goal: str) -> None:
    analysis = analyze_local_extremum(case.expression, case.start_point, goal=goal)
    gradient_x1, gradient_x2 = build_gradient_formula(case.expression)
    hessian_formula = build_hessian_formula(case.expression)

    compile_gradient_x1 = compile_objective(gradient_x1)
    compile_gradient_x2 = compile_objective(gradient_x2)
    compile_hessian = tuple(tuple(compile_objective(component) for component in row) for row in hessian_formula)

    expected_start_gradient = _expected_gradient(case.hessian, case.stationary_point, case.start_point)

    assert compile_gradient_x1(case.start_point) == pytest.approx(expected_start_gradient[0], abs=1e-8)
    assert compile_gradient_x2(case.start_point) == pytest.approx(expected_start_gradient[1], abs=1e-8)
    assert compile_gradient_x1(case.stationary_point) == pytest.approx(0.0, abs=1e-8)
    assert compile_gradient_x2(case.stationary_point) == pytest.approx(0.0, abs=1e-8)
    assert compile_hessian[0][0](case.start_point) == pytest.approx(case.hessian[0][0], abs=1e-8)
    assert compile_hessian[0][1](case.start_point) == pytest.approx(case.hessian[0][1], abs=1e-8)
    assert compile_hessian[1][0](case.start_point) == pytest.approx(case.hessian[1][0], abs=1e-8)
    assert compile_hessian[1][1](case.start_point) == pytest.approx(case.hessian[1][1], abs=1e-8)
    assert len(analysis.stationary_points) == 1
    assert analysis.stationary_points[0] == pytest.approx(case.stationary_point, abs=1e-8)


MAX_CASES = _generate_cases("max", seed=20240407, count=5)
MIN_CASES = _generate_cases("min", seed=20240408, count=5)
SADDLE_CASES = _generate_cases("saddle", seed=20240409, count=5)


@pytest.mark.parametrize("case", MAX_CASES, ids=lambda case: case.name)
def test_random_quadratic_max_cases_converge_with_gradient_ascent(case: QuadraticCase) -> None:
    _assert_analytical_block(case, goal="max")
    analysis = analyze_local_extremum(case.expression, case.start_point, goal="max")
    objective = compile_objective(case.expression)
    config = _config()

    result, metrics = run_gradient(case.expression, case.start_point, config)

    assert metrics.success is True
    assert result.success
    assert result.method_name == "gradient_ascent"
    assert len(result.records) == result.iterations_count
    assert result.iterations_count > 0
    assert any(record.step_size > 0.0 for record in result.records)
    assert analysis.stationary_point_kind == "локальный максимум"
    assert "согласуется" in analysis.goal_alignment
    assert result.optimum_point == pytest.approx(case.stationary_point, abs=5e-3)
    assert result.optimum_value == pytest.approx(objective(case.stationary_point), abs=1e-4)
    assert math.hypot(result.records[-1].gradient[0], result.records[-1].gradient[1]) <= 1e-5


@pytest.mark.parametrize("case", MAX_CASES, ids=lambda case: case.name)
def test_random_quadratic_max_cases_converge_with_conjugate_gradient(case: QuadraticCase) -> None:
    _assert_analytical_block(case, goal="max")
    analysis = analyze_local_extremum(case.expression, case.start_point, goal="max")
    objective = compile_objective(case.expression)
    config = _config()

    result, metrics = run_conjugate(case.expression, case.start_point, config)

    assert metrics.success is True
    assert result.success
    assert result.method_name == "conjugate_gradient_ascent"
    assert len(result.records) == result.iterations_count
    assert result.iterations_count > 0
    assert any(record.step_size > 0.0 for record in result.records)
    assert analysis.stationary_point_kind == "локальный максимум"
    assert "согласуется" in analysis.goal_alignment
    assert result.optimum_point == pytest.approx(case.stationary_point, abs=5e-3)
    assert result.optimum_value == pytest.approx(objective(case.stationary_point), abs=1e-4)
    assert math.hypot(result.records[-1].gradient[0], result.records[-1].gradient[1]) <= 1e-5


@pytest.mark.parametrize("case", MIN_CASES, ids=lambda case: case.name)
def test_random_quadratic_min_cases_are_classified_correctly(case: QuadraticCase) -> None:
    _assert_analytical_block(case, goal="min")

    analysis_min = analyze_local_extremum(case.expression, case.start_point, goal="min")
    analysis_max = analyze_local_extremum(case.expression, case.start_point, goal="max")

    assert analysis_min.stationary_point_kind == "локальный минимум"
    assert analysis_min.goal_alignment.startswith("Постановка на поиск минимума согласуется")
    assert analysis_max.stationary_point_kind == "локальный минимум"
    assert "не согласуется" in analysis_max.goal_alignment


@pytest.mark.parametrize("case", SADDLE_CASES, ids=lambda case: case.name)
def test_random_quadratic_saddle_cases_are_classified_correctly(case: QuadraticCase) -> None:
    _assert_analytical_block(case, goal="max")

    analysis_max = analyze_local_extremum(case.expression, case.start_point, goal="max")
    analysis_min = analyze_local_extremum(case.expression, case.start_point, goal="min")

    assert analysis_max.stationary_point_kind == "седловая точка"
    assert analysis_min.stationary_point_kind == "седловая точка"
    assert "не согласуется" in analysis_max.goal_alignment
    assert "не согласуется" in analysis_min.goal_alignment
