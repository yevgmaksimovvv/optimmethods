"""Тест контракта plugin-реестра shell для лабораторных работ."""

from __future__ import annotations

from optim_core.labs.plugins import build_default_registry


def test_default_registry_contains_expected_labs() -> None:
    registry = build_default_registry()
    items = {item.lab_id: item for item in registry.items()}

    assert set(items) == {"lr1", "lr2", "lr3"}
    assert items["lr1"].supports_embedded is True
    assert items["lr2"].supports_embedded is True
    assert items["lr3"].supports_embedded is True


def test_registry_rejects_unknown_lab() -> None:
    registry = build_default_registry()
    try:
        registry.get("unknown")
    except KeyError:
        pass
    else:
        raise AssertionError("Ожидался KeyError для незарегистрированной лабораторной")
