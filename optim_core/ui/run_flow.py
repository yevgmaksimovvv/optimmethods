"""Общий orchestrator для UI-потока применения batch-результатов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

ReportT = TypeVar("ReportT")


@dataclass(frozen=True)
class BatchRunUiHooks(Generic[ReportT]):
    """Контракт операций UI для атомарного применения отчёта."""

    assign_report: Callable[[ReportT], None]
    reset_selection: Callable[[], None]
    render_overview: Callable[[ReportT], None]
    select_first: Callable[[ReportT], bool]
    clear_details: Callable[[], None]


class BatchRunUiController(Generic[ReportT]):
    """Единый сценарий `run -> apply report -> select first/clear details`."""

    def __init__(self, hooks: BatchRunUiHooks[ReportT]) -> None:
        self._hooks = hooks

    def apply(self, report: ReportT) -> None:
        self._hooks.assign_report(report)
        self._hooks.reset_selection()
        self._hooks.render_overview(report)
        if not self._hooks.select_first(report):
            self._hooks.clear_details()
