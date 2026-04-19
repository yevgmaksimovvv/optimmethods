"""Контракт batch-контуров ЛР2 на доменные и неожиданные ошибки."""

from __future__ import annotations

from lr2.application import services as service_module
from lr2.application.services import VARIANT_PRESETS, build_polynomial, run_batch
from lr2.domain.models import BatchSummary
from lr2.domain.rosenbrock import rosenbrock_minimize as original_rosenbrock_minimize


def test_run_batch_records_domain_refusal_and_unexpected_error(monkeypatch) -> None:
    polynomial = build_polynomial("F1", VARIANT_PRESETS["variant_f1"])

    def fake_rosenbrock_minimize(*, objective, start_point, config):
        if start_point == (0.75, 0.0):
            return original_rosenbrock_minimize(objective=objective, start_point=start_point, config=config)
        if start_point == (1.0, 1.0):
            raise ValueError("Ожидаемый доменный отказ")
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(service_module, "rosenbrock_minimize", fake_rosenbrock_minimize)

    batch_result, metrics = run_batch(
        polynomial=polynomial,
        epsilons=(0.1,),
        start_points=((0.75, 0.0), (1.0, 1.0), (2.0, 2.0)),
    )

    assert metrics.total_count == 3
    assert metrics.run_count == 1
    assert metrics.failure_count == 2
    assert metrics.domain_refusal_count == 1
    assert metrics.unexpected_error_count == 1
    assert batch_result.summary == BatchSummary(
        total_count=3,
        success_count=1,
        domain_refusal_count=1,
        unexpected_error_count=1,
    )
    assert len(batch_result.runs) == 1
    assert tuple(item.status for item in batch_result.items) == (
        "success",
        "domain_refusal",
        "unexpected_error",
    )
    assert batch_result.items[1].message == "Ожидаемый доменный отказ"
    assert batch_result.items[2].message == "unexpected boom"
