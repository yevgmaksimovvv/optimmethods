"""Общие утилиты для лабораторных работ."""

from optim_core.entrypoint import GuiEntryPointSpec, run_gui_entry
from optim_core.parsing import parse_localized_float

__all__ = (
    "parse_localized_float",
    "GuiEntryPointSpec",
    "run_gui_entry",
)
