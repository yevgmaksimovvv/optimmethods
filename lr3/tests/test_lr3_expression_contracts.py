"""Контрактные тесты безопасного компилятора выражений ЛР3."""

from __future__ import annotations

import math

import pytest

from lr3.application.services import DEFAULT_CONJUGATE_EXPRESSION, DEFAULT_GRADIENT_EXPRESSION
from lr3.domain.expression import (
    ExpressionError,
    analyze_local_extremum,
    build_gradient_formula,
    build_hessian_formula,
    compile_objective,
)


def test_compile_objective_happy_path_supports_variables_functions_and_operations() -> None:
    objective = compile_objective("sin(pi / 2) + cos(0) + sqrt(abs(x1)) + pow(x2, 2) - log(e)")

    value = objective((4.0, -3.0))

    assert value == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("expression", "expected_gradient"),
    (
        (
            "x1**2 + x2**2 - x1*x2 + x1 - 2*x2",
            ("2 * x1 - x2 + 1", "2 * x2 - x1 - 2"),
        ),
        (
            "-2 - x1 - 2*x2 - 0.1*x1**2 - 100*x2**2",
            ("-1 - 0.2 * x1", "-2 - 200 * x2"),
        ),
    ),
)
def test_build_gradient_formula_returns_readable_symbolic_gradient(
    expression: str,
    expected_gradient: tuple[str, str],
) -> None:
    assert build_gradient_formula(expression) == expected_gradient


@pytest.mark.parametrize(
    ("expression", "expected_hessian"),
    (
        (
            DEFAULT_GRADIENT_EXPRESSION,
            (("2", "-1"), ("-1", "2")),
        ),
        (
            DEFAULT_CONJUGATE_EXPRESSION,
            (("-0.2", "0"), ("0", "-200")),
        ),
    ),
)
def test_build_hessian_formula_returns_readable_symbolic_matrix(
    expression: str,
    expected_hessian: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    assert build_hessian_formula(expression) == expected_hessian


def test_analyze_local_extremum_reports_quadratic_minimum_for_maximization_task() -> None:
    analysis = analyze_local_extremum(DEFAULT_GRADIENT_EXPRESSION, (0.0, 0.0), goal="max")

    assert analysis.gradient_formula == ("2 * x1 - x2 + 1", "2 * x2 - x1 - 2")
    assert analysis.gradient_at_start == pytest.approx((1.0, -2.0))
    assert len(analysis.stationary_points) == 1
    assert analysis.stationary_points[0] == pytest.approx((0.0, 1.0))
    assert analysis.stationary_gradient == pytest.approx((0.0, 0.0))
    assert analysis.hessian_at_stationary_point is not None
    assert analysis.hessian_at_stationary_point[0] == pytest.approx((2.0, -1.0))
    assert analysis.hessian_at_stationary_point[1] == pytest.approx((-1.0, 2.0))
    assert analysis.stationary_point_kind == "локальный минимум"
    assert "не согласуется" in analysis.goal_alignment
    assert "локальным минимумом" in analysis.theory_conclusion


def test_analyze_local_extremum_reports_quadratic_maximum_for_matching_task() -> None:
    analysis = analyze_local_extremum(DEFAULT_CONJUGATE_EXPRESSION, (1.0, 1.0), goal="max")

    assert analysis.gradient_formula == ("-1 - 0.2 * x1", "-2 - 200 * x2")
    assert analysis.gradient_at_start == pytest.approx((-1.2, -202.0))
    assert len(analysis.stationary_points) == 1
    assert analysis.stationary_points[0] == pytest.approx((-5.0, -0.01))
    assert analysis.stationary_gradient == pytest.approx((0.0, 0.0))
    assert analysis.hessian_at_stationary_point is not None
    assert analysis.hessian_at_stationary_point[0] == pytest.approx((-0.2, 0.0))
    assert analysis.hessian_at_stationary_point[1] == pytest.approx((0.0, -200.0))
    assert analysis.stationary_point_kind == "локальный максимум"
    assert "согласуется" in analysis.goal_alignment
    assert "локальным максимумом" in analysis.theory_conclusion


def test_analyze_local_extremum_limits_strict_conclusion_for_general_nonlinear_function() -> None:
    analysis = analyze_local_extremum("sin(x1) + x2", (0.0, 0.0), goal="max")

    assert analysis.stationary_points == ()
    assert analysis.stationary_gradient is None
    assert analysis.hessian_at_stationary_point is None
    assert analysis.stationary_point_kind == "стационарная точка не найдена"
    assert "ограничен" in analysis.strictness_note


@pytest.mark.parametrize(
    ("expression", "point", "expected"),
    (
        ("x1^2 + x2^2", (3.0, 4.0), 25.0),
        ("x1**2 + x2**2", (3.0, 4.0), 25.0),
        ("abs(-x1) + tan(0) + pi - e", (7.0, 1.0), abs(-7.0) + math.tan(0.0) + math.pi - math.e),
    ),
)
def test_compile_objective_supports_public_expression_syntax(
    expression: str,
    point: tuple[float, float],
    expected: float,
) -> None:
    objective = compile_objective(expression)

    assert objective(point) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expression", "expected_message"),
    (
        ("x1 + x3", "Неизвестный идентификатор: x3"),
        ("x1 + y", "Неизвестный идентификатор: y"),
        ("cosh(x1)", "Функция не поддерживается: cosh"),
    ),
)
def test_compile_objective_rejects_unknown_identifiers_and_functions(
    expression: str,
    expected_message: str,
) -> None:
    with pytest.raises(ExpressionError, match=expected_message):
        compile_objective(expression)


@pytest.mark.parametrize(
    ("expression", "expected_message"),
    (
        ("x1.real", "Недопустимый синтаксис: Attribute"),
        ("lambda x: x", "Недопустимый синтаксис: Lambda"),
        ("[x1][0]", "Недопустимый синтаксис: Subscript"),
    ),
)
def test_compile_objective_rejects_dangerous_syntax(expression: str, expected_message: str) -> None:
    with pytest.raises(ExpressionError, match=expected_message):
        compile_objective(expression)


@pytest.mark.parametrize(
    ("expression", "expected_message"),
    (
        ("eval('1 + 1')", "Функция не поддерживается: eval"),
        ("__import__('os')", "Функция не поддерживается: __import__"),
        ("__import__('os').system('echo 1')", "Разрешены только вызовы именованных функций."),
    ),
)
def test_compile_objective_rejects_import_like_and_eval_like_constructs(
    expression: str,
    expected_message: str,
) -> None:
    with pytest.raises(ExpressionError, match=expected_message):
        compile_objective(expression)


@pytest.mark.parametrize(
    "expression",
    (
        "",
        "x1 +",
    ),
)
def test_compile_objective_rejects_syntax_errors(expression: str) -> None:
    with pytest.raises(ExpressionError, match="Синтаксическая ошибка:"):
        compile_objective(expression)
