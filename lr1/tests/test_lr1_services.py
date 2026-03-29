"""Тесты прикладного сервиса запуска расчётов ЛР1."""

from __future__ import annotations

import pytest
from lr1.application.services import (
    build_input_config,
    parse_positive_series,
    run_batch,
)


def _build_config(method_key: str = "all"):
    return build_input_config(
        function_key="quadratic",
        kind="max",
        method_key=method_key,
        a_raw="-5",
        b_raw="5",
        eps_raw="0.01",
        l_raw="0.1",
        coefficient_raws={"a": "-2", "b": "10", "c": "3"},
    )


def test_run_batch_single_mode_works_for_one_pair() -> None:
    config = _build_config(method_key="all")

    report = run_batch(config, eps_values=(config.eps,), l_values=(config.l,))

    assert report.mode == "single"
    assert report.eps == config.eps
    assert report.l == config.l
    assert report.method_keys
    assert set(report.method_keys) == set(report.results_by_method.keys())
    for method_key in report.method_keys:
        assert len(report.grid_runs_by_method[method_key]) == 1
    assert not report.skipped_runs


def test_run_batch_grid_mode_collects_skipped_combinations() -> None:
    config = _build_config(method_key="dichotomy")

    report = run_batch(config, eps_values=(0.2, 0.01), l_values=(0.1,))

    assert report.mode == "grid"
    assert report.eps is None
    assert report.l is None
    assert report.method_keys == ("dichotomy",)
    assert len(report.grid_runs_by_method["dichotomy"]) == 1
    assert len(report.skipped_runs) == 1
    assert report.skipped_runs[0].method_key == "dichotomy"
    assert report.skipped_runs[0].eps == 0.2
    assert report.skipped_runs[0].l == 0.1


def test_parse_positive_series_parses_csv_values() -> None:
    values = parse_positive_series("0.1, 0.01, 0.001", "ε")
    assert values == (0.1, 0.01, 0.001)


def test_parse_positive_series_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="Список l пуст"):
        parse_positive_series(" ,  ", "l")
