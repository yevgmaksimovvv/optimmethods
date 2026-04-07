"""Контрактные тесты ввода и сборки конфига ЛР3."""

from __future__ import annotations

import pytest

from lr3.application.services import build_config, build_start_point, parse_int


def test_parse_int_parses_valid_integer() -> None:
    assert parse_int("17", "max_iterations") == 17


def test_parse_int_rejects_invalid_string() -> None:
    with pytest.raises(ValueError, match="Неверное целое число в поле 'max_iterations': nope"):
        parse_int("nope", "max_iterations")


def test_build_start_point_parses_coordinates() -> None:
    assert build_start_point("1,5", "-2.25") == (1.5, -2.25)


def test_build_start_point_rejects_invalid_coordinate() -> None:
    with pytest.raises(ValueError, match="Неверное число в поле 'x2': nope"):
        build_start_point("1", "nope")


def test_build_config_happy_path() -> None:
    config = build_config(
        epsilon_raw="1e-5",
        max_iterations_raw="250",
        initial_step_raw="0.1",
        timeout_raw="2.5",
        goal_raw="max",
        min_step_raw="1e-8",
        gradient_step_raw="1e-6",
        max_step_expansions_raw="16",
    )

    assert config.epsilon == 1e-5
    assert config.max_iterations == 250
    assert config.initial_step == 0.1
    assert config.timeout_seconds == 2.5
    assert config.goal == "max"
    assert config.min_step == 1e-8
    assert config.gradient_step == 1e-6
    assert config.max_step_expansions == 16


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    (
        (
            {
                "epsilon_raw": "0",
                "max_iterations_raw": "250",
                "initial_step_raw": "0.1",
                "timeout_raw": "2.5",
                "goal_raw": "max",
            },
            "epsilon должен быть > 0",
        ),
        (
            {
                "epsilon_raw": "1e-5",
                "max_iterations_raw": "0",
                "initial_step_raw": "0.1",
                "timeout_raw": "2.5",
                "goal_raw": "max",
            },
            "max_iterations должен быть > 0",
        ),
        (
            {
                "epsilon_raw": "1e-5",
                "max_iterations_raw": "250",
                "initial_step_raw": "0",
                "timeout_raw": "2.5",
                "goal_raw": "max",
            },
            "initial_step должен быть > 0",
        ),
        (
            {
                "epsilon_raw": "1e-5",
                "max_iterations_raw": "250",
                "initial_step_raw": "0.1",
                "timeout_raw": "0",
                "goal_raw": "max",
            },
            "timeout_seconds должен быть > 0",
        ),
        (
            {
                "epsilon_raw": "1e-5",
                "max_iterations_raw": "250",
                "initial_step_raw": "0.1",
                "timeout_raw": "2.5",
                "goal_raw": "sideways",
            },
            "goal должен быть min или max",
        ),
    ),
)
def test_build_config_rejects_invalid_primary_parameters(kwargs: dict[str, str], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        build_config(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    (
        (
            {
                "epsilon_raw": "1e-5",
                "max_iterations_raw": "250",
                "initial_step_raw": "0.1",
                "timeout_raw": "2.5",
                "goal_raw": "max",
                "min_step_raw": "0",
            },
            "min_step должен быть > 0",
        ),
        (
            {
                "epsilon_raw": "1e-5",
                "max_iterations_raw": "250",
                "initial_step_raw": "0.1",
                "timeout_raw": "2.5",
                "goal_raw": "max",
                "gradient_step_raw": "0",
            },
            "gradient_step должен быть > 0",
        ),
        (
            {
                "epsilon_raw": "1e-5",
                "max_iterations_raw": "250",
                "initial_step_raw": "0.1",
                "timeout_raw": "2.5",
                "goal_raw": "max",
                "max_step_expansions_raw": "0",
            },
            "max_step_expansions должен быть > 0",
        ),
    ),
)
def test_build_config_rejects_invalid_secondary_parameters(kwargs: dict[str, str], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        build_config(**kwargs)
