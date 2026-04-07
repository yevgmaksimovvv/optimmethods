"""Контрактные тесты ядра одномерного поиска ЛР1."""

from __future__ import annotations

import math

import pytest

from lr1.domain.functions import build_function_spec
from lr1.domain.search import dichotomy_search, fibonacci_search, golden_section_search


def _quadratic_minimum_spec():
    return build_function_spec(
        "quadratic",
        {
            "a": 1.0,
            "b": -4.0,
            "c": 5.0,
        },
    )


@pytest.mark.parametrize(
    "runner",
    (dichotomy_search, golden_section_search, fibonacci_search),
    ids=("dichotomy", "golden", "fibonacci"),
)
def test_search_contract_on_quadratic_minimum(runner) -> None:
    spec = _quadratic_minimum_spec()

    result = runner(spec.func, 0.0, 4.0, 0.01, 0.05, kind="min")

    assert result.method
    assert result.kind == "min"
    assert result.interval_initial == (0.0, 4.0)
    assert result.interval_final[0] <= 2.0 <= result.interval_final[1]
    assert result.interval_final[1] - result.interval_final[0] <= 0.05
    assert result.x_opt == pytest.approx(2.0, abs=0.05)
    assert result.f_opt == pytest.approx(1.0, abs=0.05)
    assert result.iterations
    assert result.func_evals > 0
    assert all(row.a <= row.lam <= row.b for row in result.iterations)
    assert all(row.a <= row.mu <= row.b for row in result.iterations)


def test_dichotomy_search_uses_safe_eval_near_pole() -> None:
    pole_spec = build_function_spec(
        "rational",
        {
            "a": 0.0,
            "b": 0.0,
            "c": 1.0,
            "d": 0.0,
            "e": 1.0,
            "f": -1.0,
        },
    )

    result = dichotomy_search(pole_spec.func, 0.85, 1.05, 0.05, 0.19, kind="max")

    assert result.interval_initial == (0.85, 1.05)
    assert result.iterations
    assert any(abs(row.lam - 1.0) < 1e-12 or abs(row.mu - 1.0) < 1e-12 for row in result.iterations)
    assert math.isfinite(result.x_opt)
    assert math.isfinite(result.f_opt)
    assert result.interval_final[0] < result.interval_final[1]
    assert result.func_evals > (2 * len(result.iterations) + 1)


@pytest.mark.parametrize(
    ("runner", "args", "expected_message"),
    (
        (dichotomy_search, (0.0, 1.0, 0.0, 0.1, "min"), "Параметр ε должен быть больше 0"),
        (dichotomy_search, (0.0, 1.0, 0.1, 0.1, "min"), "Для метода дихотомии нужно ε < l"),
        (golden_section_search, (0.0, 1.0, 0.1, 0.0, "min"), "Параметр l должен быть больше 0"),
        (fibonacci_search, (0.0, 1.0, 0.0, 0.1, "min"), "Параметр ε должен быть больше 0"),
        (fibonacci_search, (0.0, 1.0, 0.1, 0.0, "min"), "Параметр l должен быть больше 0"),
    ),
)
def test_search_rejects_invalid_parameters(runner, args, expected_message: str) -> None:
    func = _quadratic_minimum_spec().func

    with pytest.raises(ValueError, match=expected_message):
        runner(func, *args)
