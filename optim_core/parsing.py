"""Общий парсинг пользовательского ввода."""

from __future__ import annotations


def parse_localized_float(raw: str, field_name: str) -> float:
    """Парсит float со стандартной поддержкой запятой как десятичного разделителя."""
    try:
        return float(raw.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Неверное число в поле '{field_name}': {raw}") from exc
