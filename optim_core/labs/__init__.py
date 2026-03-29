"""Публичный API plugin-shell для лабораторных работ."""

from optim_core.labs.contracts import LabItemView, LabPlugin
from optim_core.labs.plugins import DefaultLabPlugin, build_default_registry
from optim_core.labs.registry import LabRegistry

__all__ = (
    "LabPlugin",
    "LabItemView",
    "LabRegistry",
    "DefaultLabPlugin",
    "build_default_registry",
)
