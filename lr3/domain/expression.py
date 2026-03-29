"""Безопасный парсинг и вычисление выражения F(x1, x2)."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from typing import TypeAlias

from lr3.domain.models import Point2D

NumericCallable: TypeAlias = Callable[..., float]
NamespaceValue: TypeAlias = float | NumericCallable

_ALLOWED_FUNCTIONS: dict[str, NumericCallable] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "pow": pow,
}
_ALLOWED_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}
_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)
_BINARY_DISPATCH: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
}
_UNARY_DISPATCH: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class ExpressionError(ValueError):
    """Ошибка в пользовательском выражении."""


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BIN_OPS):
            raise ExpressionError(f"Операция не поддерживается: {type(node.op).__name__}")
        _validate_node(node.left)
        _validate_node(node.right)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPS):
            raise ExpressionError(f"Унарная операция не поддерживается: {type(node.op).__name__}")
        _validate_node(node.operand)
        return

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("Разрешены только вызовы именованных функций.")
        if node.func.id not in _ALLOWED_FUNCTIONS:
            raise ExpressionError(f"Функция не поддерживается: {node.func.id}")
        for arg in node.args:
            _validate_node(arg)
        if node.keywords:
            raise ExpressionError("Именованные аргументы не поддерживаются.")
        return

    if isinstance(node, ast.Name):
        if node.id not in {"x1", "x2", *tuple(_ALLOWED_CONSTANTS)}:
            raise ExpressionError(f"Неизвестный идентификатор: {node.id}")
        return

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ExpressionError("Разрешены только числовые константы.")
        return

    raise ExpressionError(f"Недопустимый синтаксис: {type(node).__name__}")


def compile_objective(expression: str) -> Callable[[Point2D], float]:
    """Компилирует безопасное выражение в callable f((x1, x2))."""
    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Синтаксическая ошибка: {exc.msg}") from exc

    _validate_node(tree)
    expression_node = tree.body

    def objective(point: Point2D) -> float:
        x1, x2 = point
        namespace: dict[str, NamespaceValue] = {
            "x1": float(x1),
            "x2": float(x2),
            **_ALLOWED_CONSTANTS,
            **_ALLOWED_FUNCTIONS,
        }
        value = _evaluate_node(expression_node, namespace)
        return float(value)

    return objective


def _evaluate_node(
    node: ast.AST,
    namespace: dict[str, NamespaceValue],
) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ExpressionError("Разрешены только числовые константы.")
        return float(node.value)

    if isinstance(node, ast.Name):
        value = namespace[node.id]
        if not isinstance(value, (int, float)):
            raise ExpressionError(f"Идентификатор '{node.id}' не является числом")
        return float(value)

    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, namespace)
        right = _evaluate_node(node.right, namespace)
        binary_operation = _BINARY_DISPATCH[type(node.op)]
        return float(binary_operation(left, right))

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, namespace)
        unary_operation = _UNARY_DISPATCH[type(node.op)]
        return float(unary_operation(operand))

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = namespace[node.func.id]
        if not callable(function):
            raise ExpressionError(f"Идентификатор '{node.func.id}' не является функцией")
        arguments = [_evaluate_node(argument, namespace) for argument in node.args]
        result = function(*arguments)
        return float(result)

    raise ExpressionError(f"Недопустимый узел AST: {type(node).__name__}")
