"""Реестр плагинов лабораторных работ."""

from __future__ import annotations

from dataclasses import dataclass

from optim_core.labs.contracts import LabItemView, LabPlugin


@dataclass
class LabRegistry:
    """Хранилище зарегистрированных lab-плагинов с валидацией контрактов."""

    def __init__(self) -> None:
        self._plugins: dict[str, LabPlugin] = {}

    def register(self, plugin: LabPlugin) -> None:
        if not plugin.lab_id.strip():
            raise ValueError("lab_id не должен быть пустым")
        if plugin.lab_id in self._plugins:
            raise ValueError(f"Дубликат lab_id: {plugin.lab_id}")
        self._plugins[plugin.lab_id] = plugin

    def get(self, lab_id: str) -> LabPlugin:
        try:
            return self._plugins[lab_id]
        except KeyError as exc:
            raise KeyError(f"Плагин '{lab_id}' не зарегистрирован") from exc

    def items(self) -> tuple[LabItemView, ...]:
        return tuple(
            LabItemView(
                lab_id=plugin.lab_id,
                title=plugin.title,
                description=plugin.description,
                supports_embedded=plugin.supports_embedded,
                supports_standalone=plugin.supports_standalone,
            )
            for plugin in self._plugins.values()
        )

    def all_plugins(self) -> tuple[LabPlugin, ...]:
        return tuple(self._plugins.values())
