"""Общий парсинг пользовательского ввода."""

from __future__ import annotations


def parse_localized_float(raw: str, field_name: str) -> float:
    """Парсит float со стандартной поддержкой запятой как десятичного разделителя."""
    try:
        return float(raw.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Неверное число в поле '{field_name}': {raw}") from exc


def parse_localized_float_sequence(raw: str, field_name: str, *, separator: str = ",") -> tuple[float, ...]:
    """Парсит разделённую `separator` последовательность локализованных чисел."""
    parts = [item.strip() for item in raw.split(separator) if item.strip()]
    if not parts:
        raise ValueError(f"Список {field_name} пуст.")
    return tuple(parse_localized_float(item, field_name) for item in parts)
