"""Операции с полиномом двух переменных."""

from __future__ import annotations

from lr2.domain.models import Polynomial2D


def evaluate_polynomial(polynomial: Polynomial2D, x1: float, x2: float) -> float:
    """Вычисляет значение полинома в точке (x1, x2)."""
    result = 0.0
    for i, row in enumerate(polynomial.coefficients):
        x1_power = x1**i
        for j, coefficient in enumerate(row):
            if coefficient == 0.0:
                continue
            result += coefficient * x1_power * (x2**j)
    return result


def format_polynomial(polynomial: Polynomial2D) -> str:
    """Строит текст формулы по матрице коэффициентов."""
    terms: list[str] = []
    for i, row in enumerate(polynomial.coefficients):
        for j, coefficient in enumerate(row):
            if abs(coefficient) < 1e-14:
                continue
            sign = " + " if coefficient >= 0 else " - "
            abs_coeff = abs(coefficient)
            coeff_part = f"{abs_coeff:.6g}"
            powers: list[str] = []
            if i > 0:
                powers.append("x1" if i == 1 else f"x1^{i}")
            if j > 0:
                powers.append("x2" if j == 1 else f"x2^{j}")
            power_part = "*".join(powers)
            term = coeff_part if not power_part else f"{coeff_part}*{power_part}"
            if not terms:
                terms.append(term if coefficient >= 0 else f"-{term}")
            else:
                terms.append(f"{sign}{term}")
    return "".join(terms) if terms else "0"
