"""Безопасный парсинг и вычисление выражения F(x1, x2)."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from typing import TypeAlias

from lr3.domain.models import ExtremumAnalysis, Hessian2D, Point2D

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


def build_gradient_formula(expression: str) -> tuple[str, str]:
    """Строит символьные компоненты градиента для отображения в UI."""
    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Синтаксическая ошибка: {exc.msg}") from exc

    _validate_node(tree)
    expression_node = tree.body
    gradient_x1 = _format_expression(_simplify(_differentiate(expression_node, "x1")))
    gradient_x2 = _format_expression(_simplify(_differentiate(expression_node, "x2")))
    return gradient_x1, gradient_x2


def build_hessian_formula(expression: str) -> Hessian2D:
    """Строит символьную матрицу Гессе для двумерной функции."""
    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Синтаксическая ошибка: {exc.msg}") from exc

    _validate_node(tree)
    expression_node = tree.body
    h11 = _format_expression(_simplify(_differentiate(_differentiate(expression_node, "x1"), "x1")))
    h12 = _format_expression(_simplify(_differentiate(_differentiate(expression_node, "x1"), "x2")))
    h21 = _format_expression(_simplify(_differentiate(_differentiate(expression_node, "x2"), "x1")))
    h22 = _format_expression(_simplify(_differentiate(_differentiate(expression_node, "x2"), "x2")))
    return ((h11, h12), (h21, h22))


def analyze_local_extremum(
    expression: str,
    start_point: Point2D,
    goal: str,
) -> ExtremumAnalysis:
    """Даёт честный учебный анализ функции перед запуском метода."""
    gradient_formula = build_gradient_formula(expression)
    gradient_at_start = (
        compile_objective(gradient_formula[0])(start_point),
        compile_objective(gradient_formula[1])(start_point),
    )
    hessian_formula = build_hessian_formula(expression)
    stationary_points, stationary_gradient, stationary_hessian = _solve_stationary_point(
        gradient_formula=gradient_formula,
        hessian_formula=hessian_formula,
    )
    stationary_point_kind = _classify_stationary_point(stationary_points, stationary_hessian)
    theory_conclusion = _build_theory_conclusion(stationary_point_kind, stationary_points, stationary_hessian)
    goal_alignment = _build_goal_alignment(goal, stationary_point_kind)
    strictness_note = _build_strictness_note(stationary_points, stationary_hessian)
    return ExtremumAnalysis(
        expression=expression,
        start_point=start_point,
        goal=goal,
        gradient_formula=gradient_formula,
        gradient_at_start=gradient_at_start,
        hessian_formula=hessian_formula,
        stationary_points=stationary_points,
        stationary_gradient=stationary_gradient,
        hessian_at_stationary_point=stationary_hessian,
        stationary_point_kind=stationary_point_kind,
        theory_conclusion=theory_conclusion,
        goal_alignment=goal_alignment,
        strictness_note=strictness_note,
    )


def _differentiate(node: ast.AST, variable: str) -> ast.AST:
    if isinstance(node, ast.Constant):
        return _number(0)

    if isinstance(node, ast.Name):
        if node.id == variable:
            return _number(1)
        if node.id in {"x1", "x2"} | set(_ALLOWED_CONSTANTS):
            return _number(0)
        raise ExpressionError(f"Неизвестный идентификатор: {node.id}")

    if isinstance(node, ast.UnaryOp):
        operand = _differentiate(node.operand, variable)
        if isinstance(node.op, ast.UAdd):
            return operand
        return ast.UnaryOp(op=ast.USub(), operand=operand)

    if isinstance(node, ast.BinOp):
        left = node.left
        right = node.right
        if isinstance(node.op, ast.Add):
            return ast.BinOp(left=_differentiate(left, variable), op=ast.Add(), right=_differentiate(right, variable))
        if isinstance(node.op, ast.Sub):
            return ast.BinOp(left=_differentiate(left, variable), op=ast.Sub(), right=_differentiate(right, variable))
        if isinstance(node.op, ast.Mult):
            return ast.BinOp(
                left=ast.BinOp(left=_differentiate(left, variable), op=ast.Mult(), right=right),
                op=ast.Add(),
                right=ast.BinOp(left=left, op=ast.Mult(), right=_differentiate(right, variable)),
            )
        if isinstance(node.op, ast.Div):
            numerator = ast.BinOp(
                left=ast.BinOp(left=_differentiate(left, variable), op=ast.Mult(), right=right),
                op=ast.Sub(),
                right=ast.BinOp(left=left, op=ast.Mult(), right=_differentiate(right, variable)),
            )
            denominator = ast.BinOp(left=right, op=ast.Pow(), right=_number(2))
            return ast.BinOp(left=numerator, op=ast.Div(), right=denominator)
        if isinstance(node.op, ast.Pow):
            return _differentiate_power(left, right, variable)
        if isinstance(node.op, ast.Mod):
            raise ExpressionError("Дифференцирование операции % не поддерживается")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if len(node.args) != 1:
            if name == "pow" and len(node.args) == 2:
                return _differentiate_power(node.args[0], node.args[1], variable)
            raise ExpressionError(f"Дифференцирование функции {name} не поддерживается")

        argument = node.args[0]
        d_argument = _differentiate(argument, variable)
        if name == "sin":
            return ast.BinOp(
                left=ast.Call(func=ast.Name(id="cos", ctx=ast.Load()), args=[argument], keywords=[]),
                op=ast.Mult(),
                right=d_argument,
            )
        if name == "cos":
            return ast.BinOp(
                left=ast.UnaryOp(
                    op=ast.USub(),
                    operand=ast.Call(func=ast.Name(id="sin", ctx=ast.Load()), args=[argument], keywords=[]),
                ),
                op=ast.Mult(),
                right=d_argument,
            )
        if name == "tan":
            cosine = ast.Call(func=ast.Name(id="cos", ctx=ast.Load()), args=[argument], keywords=[])
            denominator = ast.BinOp(left=cosine, op=ast.Pow(), right=_number(2))
            return ast.BinOp(left=d_argument, op=ast.Div(), right=denominator)
        if name == "exp":
            return ast.BinOp(
                left=ast.Call(func=ast.Name(id="exp", ctx=ast.Load()), args=[argument], keywords=[]),
                op=ast.Mult(),
                right=d_argument,
            )
        if name == "log":
            return ast.BinOp(left=d_argument, op=ast.Div(), right=argument)
        if name == "sqrt":
            denominator = ast.BinOp(
                left=_number(2),
                op=ast.Mult(),
                right=ast.Call(func=ast.Name(id="sqrt", ctx=ast.Load()), args=[argument], keywords=[]),
            )
            return ast.BinOp(left=d_argument, op=ast.Div(), right=denominator)
        if name == "abs":
            ratio = ast.BinOp(left=argument, op=ast.Div(), right=ast.Call(func=ast.Name(id="abs", ctx=ast.Load()), args=[argument], keywords=[]))
            return ast.BinOp(left=ratio, op=ast.Mult(), right=d_argument)
        if name == "pow":
            return _differentiate_power(node.args[0], node.args[1], variable)
        raise ExpressionError(f"Дифференцирование функции {name} не поддерживается")

    raise ExpressionError(f"Дифференцирование узла {type(node).__name__} не поддерживается")


def _differentiate_power(base: ast.AST, exponent: ast.AST, variable: str) -> ast.AST:
    d_base = _differentiate(base, variable)
    d_exponent = _differentiate(exponent, variable)

    if _is_zero(d_exponent):
        if isinstance(exponent, ast.Constant) and isinstance(exponent.value, (int, float)):
            if exponent.value == 0:
                return _number(0)
            if exponent.value == 1:
                return d_base
            new_exponent = _number(float(exponent.value) - 1.0)
            return ast.BinOp(
                left=ast.BinOp(left=exponent, op=ast.Mult(), right=ast.BinOp(left=base, op=ast.Pow(), right=new_exponent)),
                op=ast.Mult(),
                right=d_base,
            )

    power_term = ast.BinOp(left=base, op=ast.Pow(), right=exponent)
    log_term = ast.BinOp(left=d_exponent, op=ast.Mult(), right=ast.Call(func=ast.Name(id="log", ctx=ast.Load()), args=[base], keywords=[]))
    base_term = ast.BinOp(
        left=exponent,
        op=ast.Mult(),
        right=ast.BinOp(left=d_base, op=ast.Div(), right=base),
    )
    return ast.BinOp(
        left=power_term,
        op=ast.Mult(),
        right=ast.BinOp(left=log_term, op=ast.Add(), right=base_term),
    )


def _solve_stationary_point(
    *,
    gradient_formula: tuple[str, str],
    hessian_formula: Hessian2D,
) -> tuple[tuple[Point2D, ...], Point2D | None, Hessian2D | None]:
    if not _is_constant_matrix(hessian_formula):
        return (), None, None

    hessian = _evaluate_matrix(hessian_formula, (0.0, 0.0))
    determinant = _determinant_2x2(hessian)
    if abs(determinant) <= 1e-12:
        return (), None, hessian

    gradient_at_zero = (
        compile_objective(gradient_formula[0])((0.0, 0.0)),
        compile_objective(gradient_formula[1])((0.0, 0.0)),
    )
    rhs = (-gradient_at_zero[0], -gradient_at_zero[1])
    stationary_point = _solve_linear_system_2x2(hessian, rhs)
    stationary_gradient = (
        compile_objective(gradient_formula[0])(stationary_point),
        compile_objective(gradient_formula[1])(stationary_point),
    )
    return (stationary_point,), stationary_gradient, hessian


def _classify_stationary_point(
    stationary_points: tuple[Point2D, ...],
    hessian: Hessian2D | None,
) -> str:
    if not stationary_points:
        return "стационарная точка не найдена"
    if hessian is None:
        return "строгий аналитический вывод недоступен"

    a11 = hessian[0][0]
    a12 = hessian[0][1]
    a21 = hessian[1][0]
    a22 = hessian[1][1]
    if abs(a12 - a21) > 1e-8:
        return "строгий аналитический вывод недоступен"

    determinant = a11 * a22 - a12 * a21
    if determinant > 1e-12 and a11 > 0:
        return "локальный минимум"
    if determinant > 1e-12 and a11 < 0:
        return "локальный максимум"
    if determinant < -1e-12:
        return "седловая точка"
    return "строгий аналитический вывод недоступен"


def _build_theory_conclusion(
    stationary_point_kind: str,
    stationary_points: tuple[Point2D, ...],
    hessian: Hessian2D | None,
) -> str:
    if not stationary_points:
        return "Стационарную точку в рамках текущей модели аналитически выделить не удалось."
    if stationary_point_kind == "локальный минимум":
        return "Стационарная точка является локальным минимумом по положительной определённости матрицы Гессе."
    if stationary_point_kind == "локальный максимум":
        return "Стационарная точка является локальным максимумом по отрицательной определённости матрицы Гессе."
    if stationary_point_kind == "седловая точка":
        return "Стационарная точка является седловой, поэтому искомый экстремум не подтверждается."
    if hessian is None:
        return "Матрица Гессе не даёт строгого вывода в текущем классе функции."
    return "Матрица Гессе не позволяет сделать строгий вывод о характере стационарной точки."


def _build_goal_alignment(goal: str, stationary_point_kind: str) -> str:
    if goal == "max":
        goal_label = "максимума"
    elif goal == "min":
        goal_label = "минимума"
    else:
        goal_label = "экстремума"
    if goal == "max" and stationary_point_kind == "локальный максимум":
        return f"Постановка на поиск {goal_label} согласуется с аналитическим выводом."
    if goal == "min" and stationary_point_kind == "локальный минимум":
        return f"Постановка на поиск {goal_label} согласуется с аналитическим выводом."
    if stationary_point_kind in {"локальный максимум", "локальный минимум", "седловая точка"}:
        return f"Постановка на поиск {goal_label} не согласуется с аналитическим выводом."
    return f"Постановка на поиск {goal_label} не подтверждена строгим аналитическим выводом."


def _build_strictness_note(
    stationary_points: tuple[Point2D, ...],
    hessian: Hessian2D | None,
) -> str:
    if stationary_points and hessian is not None:
        return "Для этой функции строгий вывод получен по системе ∇f(x) = 0 и матрице Гессе."
    return "Для произвольной пользовательской функции строгий вывод ограничен случаями, где система ∇f(x) = 0 и матрица Гессе допускают прямой анализ."


def _solve_linear_system_2x2(matrix: Hessian2D, rhs: Point2D) -> Point2D:
    a11, a12 = matrix[0]
    a21, a22 = matrix[1]
    determinant = _determinant_2x2(matrix)
    x1 = (rhs[0] * a22 - a12 * rhs[1]) / determinant
    x2 = (a11 * rhs[1] - rhs[0] * a21) / determinant
    return (x1, x2)


def _determinant_2x2(matrix: Hessian2D) -> float:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _is_constant_matrix(matrix: Hessian2D) -> bool:
    return all(_is_constant_expression(component) for row in matrix for component in row)


def _is_constant_expression(expression: str) -> bool:
    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return False
    return _is_constant_ast(tree.body)


def _is_constant_ast(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.UnaryOp):
        return _is_constant_ast(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_constant_ast(node.left) and _is_constant_ast(node.right)
    if isinstance(node, ast.Call):
        return all(_is_constant_ast(arg) for arg in node.args)
    return False


def _evaluate_matrix(matrix: Hessian2D, point: Point2D) -> Hessian2D:
    return (
        (compile_objective(matrix[0][0])(point), compile_objective(matrix[0][1])(point)),
        (compile_objective(matrix[1][0])(point), compile_objective(matrix[1][1])(point)),
    )


def _number(value: float) -> ast.Constant:
    rounded = round(value)
    if math.isfinite(value) and abs(value - rounded) < 1e-12:
        return ast.Constant(value=int(rounded))
    return ast.Constant(value=float(value))


def _is_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and float(node.value) == 0.0


def _is_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and float(node.value) == 1.0


def _is_number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _simplify(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.Constant):
        return _number(float(node.value))

    if isinstance(node, ast.Name):
        return node

    if isinstance(node, ast.UnaryOp):
        operand = _simplify(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
            return _number(-float(operand.value))
        if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
            return _simplify(operand.operand)
        return ast.UnaryOp(op=ast.USub(), operand=operand)

    if isinstance(node, ast.BinOp):
        left = _simplify(node.left)
        right = _simplify(node.right)

        if _is_number(left) and _is_number(right):
            if isinstance(node.op, ast.Add):
                return _number(float(left.value) + float(right.value))
            if isinstance(node.op, ast.Sub):
                return _number(float(left.value) - float(right.value))
            if isinstance(node.op, ast.Mult):
                return _number(float(left.value) * float(right.value))
            if isinstance(node.op, ast.Div):
                return _number(float(left.value) / float(right.value))
            if isinstance(node.op, ast.Pow):
                return _number(float(left.value) ** float(right.value))
            if isinstance(node.op, ast.Mod):
                return _number(float(left.value) % float(right.value))

        if isinstance(node.op, ast.Add):
            if _is_zero(left):
                return right
            if _is_zero(right):
                return left
            return ast.BinOp(left=left, op=ast.Add(), right=right)
        if isinstance(node.op, ast.Sub):
            if _is_zero(right):
                return left
            if _is_zero(left):
                return ast.UnaryOp(op=ast.USub(), operand=right)
            return ast.BinOp(left=left, op=ast.Sub(), right=right)
        if isinstance(node.op, ast.Mult):
            factors = _flatten_product(left) + _flatten_product(right)
            constant = 1.0
            non_constant_factors: list[ast.AST] = []
            for factor in factors:
                if _is_number(factor):
                    constant *= float(factor.value)
                else:
                    non_constant_factors.append(factor)
            if constant == 0.0:
                return _number(0)
            if constant != 1.0 or not non_constant_factors:
                non_constant_factors.insert(0, _number(constant))
            return _rebuild_product(non_constant_factors)
        if isinstance(node.op, ast.Div):
            if _is_zero(left):
                return _number(0)
            if _is_one(right):
                return left
            return ast.BinOp(left=left, op=ast.Div(), right=right)
        if isinstance(node.op, ast.Pow):
            if _is_zero(right):
                return _number(1)
            if _is_one(right):
                return left
            if _is_zero(left):
                return _number(0)
            return ast.BinOp(left=left, op=ast.Pow(), right=right)
        if isinstance(node.op, ast.Mod):
            return ast.BinOp(left=left, op=ast.Mod(), right=right)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        simplified_args = [_simplify(arg) for arg in node.args]
        if all(_is_number(arg) for arg in simplified_args):
            namespace: dict[str, NamespaceValue] = {
                **_ALLOWED_CONSTANTS,
                **_ALLOWED_FUNCTIONS,
            }
            function = namespace.get(node.func.id)
            if callable(function):
                values = [float(arg.value) for arg in simplified_args]
                return _number(float(function(*values)))
        return ast.Call(func=ast.Name(id=node.func.id, ctx=ast.Load()), args=simplified_args, keywords=[])

    return node


def _flatten_product(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _flatten_product(node.left) + _flatten_product(node.right)
    return [node]


def _rebuild_product(factors: list[ast.AST]) -> ast.AST:
    if not factors:
        return _number(1)
    expression = factors[0]
    for factor in factors[1:]:
        expression = ast.BinOp(left=expression, op=ast.Mult(), right=factor)
    return expression


def _format_expression(node: ast.AST) -> str:
    expression = ast.Expression(body=node)
    ast.fix_missing_locations(expression)
    return ast.unparse(expression.body if hasattr(ast, "unparse") else expression)


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
