"""Контрактные тесты безопасного компилятора выражений ЛР3."""

from __future__ import annotations

import math

import pytest

from lr3.domain.expression import ExpressionError, compile_objective


def test_compile_objective_happy_path_supports_variables_functions_and_operations() -> None:
    objective = compile_objective("sin(pi / 2) + cos(0) + sqrt(abs(x1)) + pow(x2, 2) - log(e)")

    value = objective((4.0, -3.0))

    assert value == pytest.approx(12.0)


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
