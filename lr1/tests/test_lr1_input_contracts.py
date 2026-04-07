"""Контрактные тесты ввода и сборки конфига ЛР1."""

from __future__ import annotations

import pytest

from lr1.application.services import build_input_config, parse_positive_series


def test_parse_positive_series_parses_csv_values() -> None:
    values = parse_positive_series("0.1, 0.01, 1e-3", "ε")

    assert values == (0.1, 0.01, 0.001)


@pytest.mark.parametrize(
    ("raw", "field_name", "expected_message"),
    (
        (" , ", "l", "Список l пуст."),
        ("0.1, -0.2", "ε", "Все значения ε должны быть > 0."),
        ("0.1, foo", "ε", "Неверное число в поле 'ε': foo"),
    ),
)
def test_parse_positive_series_rejects_invalid_input(
    raw: str,
    field_name: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        parse_positive_series(raw, field_name)


def test_build_input_config_happy_path_for_quadratic() -> None:
    config = build_input_config(
        function_key="quadratic",
        kind="max",
        method_key="all",
        a_raw="-5",
        b_raw="5",
        eps_raw="0.01",
        l_raw="0.1",
        coefficient_raws={"a": "-2", "b": "10", "c": "3"},
    )

    assert config.function_spec.key == "quadratic"
    assert config.kind == "max"
    assert config.method_key == "all"
    assert config.interval_raw == (-5.0, 5.0)
    assert config.interval == (-5.0, 5.0)
    assert config.eps == 0.01
    assert config.l == 0.1
    assert config.function_spec.coefficient_values == (("a", -2.0), ("b", 10.0), ("c", 3.0))


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    (
        (
            {
                "function_key": "unknown",
                "kind": "max",
                "method_key": "all",
                "a_raw": "-5",
                "b_raw": "5",
                "eps_raw": "0.01",
                "l_raw": "0.1",
                "coefficient_raws": {"a": "-2", "b": "10", "c": "3"},
            },
            "Неизвестная функция: unknown",
        ),
        (
            {
                "function_key": "quadratic",
                "kind": "max",
                "method_key": "bad-method",
                "a_raw": "-5",
                "b_raw": "5",
                "eps_raw": "0.01",
                "l_raw": "0.1",
                "coefficient_raws": {"a": "-2", "b": "10", "c": "3"},
            },
            "Неизвестный метод: bad-method",
        ),
        (
            {
                "function_key": "quadratic",
                "kind": "max",
                "method_key": "all",
                "a_raw": "5",
                "b_raw": "5",
                "eps_raw": "0.01",
                "l_raw": "0.1",
                "coefficient_raws": {"a": "-2", "b": "10", "c": "3"},
            },
            "Должно быть a < b.",
        ),
        (
            {
                "function_key": "quadratic",
                "kind": "max",
                "method_key": "all",
                "a_raw": "-5",
                "b_raw": "5",
                "eps_raw": "0",
                "l_raw": "0.1",
                "coefficient_raws": {"a": "-2", "b": "10", "c": "3"},
            },
            "Параметры ε и l должны быть положительными.",
        ),
        (
            {
                "function_key": "quadratic",
                "kind": "max",
                "method_key": "all",
                "a_raw": "-5",
                "b_raw": "5",
                "eps_raw": "0.01",
                "l_raw": "-0.1",
                "coefficient_raws": {"a": "-2", "b": "10", "c": "3"},
            },
            "Параметры ε и l должны быть положительными.",
        ),
    ),
)
def test_build_input_config_rejects_invalid_core_contract(
    kwargs: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        build_input_config(**kwargs)


def test_build_input_config_shifts_interval_at_forbidden_point_boundary() -> None:
    config = build_input_config(
        function_key="rational",
        kind="min",
        method_key="golden",
        a_raw="2",
        b_raw="3",
        eps_raw="0.01",
        l_raw="0.1",
        coefficient_raws={},
    )

    assert config.interval_raw == (2.0, 3.0)
    assert config.interval[0] > config.interval_raw[0]
    assert config.interval[1] == pytest.approx(config.interval_raw[1])
