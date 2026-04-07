"""Состояние UI для ЛР1."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from lr1.domain.models import RunReport

logger = logging.getLogger("lr1.ui.state")


@dataclass
class AppState:
    """Минимальное состояние GUI между действиями пользователя."""

    last_report: Optional[RunReport] = None
    busy: bool = False
    selected_table_method: str = ""
    selected_grid_run_index: int = 0

    def __post_init__(self) -> None:
        logger.debug(
            "AppState created busy=%s selected_table_method=%s selected_grid_run_index=%d",
            self.busy,
            self.selected_table_method,
            self.selected_grid_run_index,
        )
