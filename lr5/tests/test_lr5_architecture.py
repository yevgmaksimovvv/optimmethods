"""Проверка архитектурных границ для ЛР5."""

from __future__ import annotations

from pathlib import Path

from optim_core.testing.architecture import find_forbidden_import_offenders

DOMAIN_DIR = Path(__file__).resolve().parents[1] / "domain"
FORBIDDEN_IMPORT_PREFIXES = ("PySide6", "matplotlib")


def test_domain_layer_has_no_ui_or_plot_dependencies() -> None:
    offenders = find_forbidden_import_offenders(DOMAIN_DIR, FORBIDDEN_IMPORT_PREFIXES)
    assert not offenders, f"Нарушены границы доменного слоя: {offenders}"
