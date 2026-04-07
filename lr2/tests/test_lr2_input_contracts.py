"""Контрактные тесты ввода и сборки данных ЛР2."""

from __future__ import annotations

import pytest

from lr2.domain.models import normalize_coefficients
from lr2.application.services import parse_epsilons, parse_points


def test_parse_epsilons_parses_csv_values() -> None:
    values = parse_epsilons("0.1, 1e-2, 0.005")

    assert values == (0.1, 0.01, 0.005)


@pytest.mark.parametrize(
    ("raw", "expected_message"),
    (
        (" , ", "Список epsilon пуст."),
        ("0.1, 0, 0.2", "Все epsilon должны быть > 0."),
        ("0.1, nope", "Неверное число в поле 'epsilon': nope"),
    ),
)
def test_parse_epsilons_rejects_invalid_input(raw: str, expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        parse_epsilons(raw)


def test_parse_points_parses_multiline_point_list() -> None:
    points = parse_points("1;2 | 3;4\n5;6")

    assert points == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))


@pytest.mark.parametrize(
    ("raw", "expected_message"),
    (
        (" ", "Список стартовых точек пуст."),
        ("1;2;3", "Неверный формат стартовой точки."),
    ),
)
def test_parse_points_rejects_invalid_input(raw: str, expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        parse_points(raw)


def test_normalize_coefficients_returns_rectangular_float_matrix() -> None:
    matrix = normalize_coefficients(((1, 2, 3), (4.5, 0, -6)))

    assert matrix == ((1.0, 2.0, 3.0), (4.5, 0.0, -6.0))


@pytest.mark.parametrize(
    ("matrix", "expected_message"),
    (
        ((), "Матрица коэффициентов пуста."),
        (((1, 2), (3,)), "Матрица коэффициентов должна быть прямоугольной."),
        (((),), "Матрица коэффициентов содержит пустую строку."),
    ),
)
def test_normalize_coefficients_rejects_invalid_shape(matrix, expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        normalize_coefficients(matrix)
