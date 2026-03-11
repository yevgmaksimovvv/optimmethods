"""Построение функций, доступных в лабораторной работе.

Модуль отвечает за две задачи:
- собирать вычислимую функцию из набора коэффициентов;
- заранее извлекать полезную аналитику: стационарные точки и точки разрыва.

Благодаря этому UI и сервисы работают не с "сырыми коэффициентами",
а с объектом `FunctionSpec`, в котором уже есть всё нужное для расчёта
и пояснений пользователю.
"""

import logging
import math
from typing import Dict, Iterable, Tuple

from lr1.domain.models import CoefficientSpec, FunctionSpec, FunctionTemplateSpec
from lr1.domain.numerical import far_from_all, unique_sorted
from lr1.infrastructure.logging import configure_logging
from lr1.infrastructure.settings import DENOMINATOR_TOLERANCE, POLYNOMIAL_TOLERANCE


configure_logging()
logger = logging.getLogger("lr1.function_defs")


def _format_number(value: float) -> str:
    """Красиво форматирует число для подписи формулы."""
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.6g}"


def _format_signed_term(value: float, suffix: str, is_first: bool = False) -> str:
    """Формирует одно слагаемое полинома с правильным знаком."""
    if abs(value) < POLYNOMIAL_TOLERANCE:
        return ""

    abs_value = abs(value)
    coeff = "" if suffix and abs(abs_value - 1.0) < POLYNOMIAL_TOLERANCE else _format_number(abs_value)
    term = f"{coeff}{suffix}"

    if is_first:
        return term if value >= 0 else f"-{term}"
    sign = " + " if value >= 0 else " - "
    return f"{sign}{term}"


def _format_polynomial(a: float, b: float, c: float) -> str:
    """Собирает строковое представление квадратного трёхчлена."""
    parts = [
        _format_signed_term(a, "x²", True),
        _format_signed_term(b, "x"),
        _format_signed_term(c, ""),
    ]
    text = "".join(part for part in parts if part)
    return text or "0"


def solve_real_roots(a: float, b: float, c: float) -> Tuple[float, ...]:
    """Находит действительные корни квадратного или линейного уравнения."""
    if abs(a) < POLYNOMIAL_TOLERANCE:
        if abs(b) < POLYNOMIAL_TOLERANCE:
            return ()
        return (-c / b,)

    discriminant = (b * b) - (4.0 * a * c)
    if discriminant < -POLYNOMIAL_TOLERANCE:
        return ()
    if abs(discriminant) <= POLYNOMIAL_TOLERANCE:
        return (-b / (2.0 * a),)

    sqrt_d = math.sqrt(discriminant)
    roots = (
        (-b - sqrt_d) / (2.0 * a),
        (-b + sqrt_d) / (2.0 * a),
    )
    return unique_sorted(roots)


def _build_quadratic_spec(coefficients: Dict[str, float]) -> FunctionSpec:
    """Создаёт спецификацию квадратичной функции и её стационарной точки."""
    a = coefficients["a"]
    b = coefficients["b"]
    c = coefficients["c"]

    def func(x: float) -> float:
        return (a * x * x) + (b * x) + c

    stationary_points = solve_real_roots(0.0, 2.0 * a, b)
    title = f"f(x) = {_format_polynomial(a, b, c)}"
    return FunctionSpec(
        key="quadratic",
        title=title,
        func=func,
        forbidden_points=(),
        stationary_points=stationary_points,
        coefficient_values=tuple(coefficients.items()),
        formula_hint="f(x) = a·x² + b·x + c",
    )


def _build_rational_spec(coefficients: Dict[str, float]) -> FunctionSpec:
    """Создаёт спецификацию рациональной функции.

    Помимо самой функции, вычисляет:
    - действительные корни знаменателя как точки разрыва;
    - стационарные точки, которые могут быть кандидатами на экстремум.
    """
    a = coefficients["a"]
    b = coefficients["b"]
    c = coefficients["c"]
    d = coefficients["d"]
    e = coefficients["e"]
    f = coefficients["f"]

    if abs(d) < POLYNOMIAL_TOLERANCE and abs(e) < POLYNOMIAL_TOLERANCE and abs(f) < POLYNOMIAL_TOLERANCE:
        raise ValueError("Знаменатель не может быть тождественно равен нулю.")

    forbidden_points = solve_real_roots(d, e, f)
    derivative_roots = solve_real_roots((a * e) - (b * d), 2.0 * ((a * f) - (c * d)), (b * f) - (c * e))
    stationary_points = tuple(root for root in derivative_roots if far_from_all(root, forbidden_points))

    def func(x: float) -> float:
        den = (d * x * x) + (e * x) + f
        if abs(den) < DENOMINATOR_TOLERANCE:
            logger.warning("Rational function undefined at x=%s due to near-zero denominator=%s", x, den)
            raise ZeroDivisionError(f"Function is undefined at x={x}")
        num = (a * x * x) + (b * x) + c
        return num / den

    num_text = _format_polynomial(a, b, c)
    den_text = _format_polynomial(d, e, f)
    title = f"f(x) = ({num_text}) / ({den_text})"
    return FunctionSpec(
        key="rational",
        title=title,
        func=func,
        forbidden_points=forbidden_points,
        stationary_points=stationary_points,
        coefficient_values=tuple(coefficients.items()),
        formula_hint="f(x) = (a·x² + b·x + c) / (d·x² + e·x + f)",
    )


FUNCTION_TEMPLATE_SPECS = {
    "quadratic": FunctionTemplateSpec(
        key="quadratic",
        title="Квадратичная",
        formula_hint="f(x) = a·x² + b·x + c",
        coefficients=(
            CoefficientSpec("a", "a", -2.0),
            CoefficientSpec("b", "b", 10.0),
            CoefficientSpec("c", "c", 3.0),
        ),
        builder=_build_quadratic_spec,
    ),
    "rational": FunctionTemplateSpec(
        key="rational",
        title="Рациональная",
        formula_hint="f(x) = (a·x² + b·x + c) / (d·x² + e·x + f)",
        coefficients=(
            CoefficientSpec("a", "a", 2.0),
            CoefficientSpec("b", "b", 0.0),
            CoefficientSpec("c", "c", 3.0),
            CoefficientSpec("d", "d", 1.0),
            CoefficientSpec("e", "e", 2.0),
            CoefficientSpec("f", "f", -8.0),
        ),
        builder=_build_rational_spec,
    ),
}


def build_function_spec(function_key: str, coefficients: Dict[str, float]) -> FunctionSpec:
    """Строит `FunctionSpec` по выбранному типу функции и коэффициентам."""
    if function_key not in FUNCTION_TEMPLATE_SPECS:
        raise ValueError(f"Неизвестная функция: {function_key}")
    return FUNCTION_TEMPLATE_SPECS[function_key].builder(coefficients)


def analytic_comment(function_spec: FunctionSpec, interval: Tuple[float, float], kind: str) -> str:
    """Формирует короткое текстовое пояснение по теории для выбранной функции.

    Этот текст не влияет на расчёт. Он нужен только для интерфейса,
    чтобы пользователь видел, откуда вообще берутся кандидаты на экстремум.
    """
    logger.debug("analytic_comment function=%s interval=%s kind=%s", function_spec.key, interval, kind)
    a, b = interval
    lines = []

    inner_breaks = [point for point in function_spec.forbidden_points if a < point < b]
    if inner_breaks:
        points = ", ".join(f"{point:.6f}" for point in inner_breaks)
        lines.append(f"На выбранном интервале есть точки разрыва: {points}.")
    elif function_spec.forbidden_points:
        points = ", ".join(f"{point:.6f}" for point in function_spec.forbidden_points)
        lines.append(f"Точки разрыва функции: {points}.")

    inner_stationary = [point for point in function_spec.stationary_points if a < point < b]
    if inner_stationary:
        points = ", ".join(f"{point:.6f}" for point in inner_stationary)
        lines.append(f"Внутренние стационарные точки на интервале: {points}.")
    else:
        lines.append("На выбранном интервале нет внутренних стационарных точек.")

    if kind == "max":
        lines.append("Теоретический ориентир ищется среди граничных и стационарных точек.")
    else:
        lines.append("Для минимума проверяются границы интервала и внутренние стационарные точки.")

    comment = "\n".join(lines)
    logger.debug("analytic_comment result function=%s text=%s", function_spec.key, comment)
    return comment


logger.debug("Function templates loaded. available_functions=%s", tuple(FUNCTION_TEMPLATE_SPECS))
