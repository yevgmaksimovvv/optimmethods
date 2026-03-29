"""Общие helpers для matplotlib-графиков в Qt."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget

DARK_MATPLOTLIB_RC = {
    "figure.facecolor": "#171b24",
    "axes.facecolor": "#10141f",
    "axes.edgecolor": "#4a5974",
    "axes.labelcolor": "#dce6f5",
    "axes.titlecolor": "#e8f0ff",
    "xtick.color": "#c4cfdf",
    "ytick.color": "#c4cfdf",
    "grid.color": "#2c3b55",
    "grid.alpha": 0.35,
    "text.color": "#dce6f5",
    "legend.facecolor": "#161d2b",
    "legend.edgecolor": "#4a5974",
}


class PlotCanvas(FigureCanvasQTAgg):
    """Единый canvas для вкладок графиков ЛР."""

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        figsize: tuple[float, float] = (13.0, 8.0),
        dpi: int = 100,
    ) -> None:
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.figure.set_constrained_layout(True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.patch.set_facecolor("#171b24")


def clear_plot_canvas(canvas: PlotCanvas, message: str) -> None:
    """Сбрасывает график в единое пустое состояние."""
    canvas.figure.clear()
    canvas.figure.patch.set_facecolor("#171b24")
    axis = canvas.figure.add_subplot(1, 1, 1)
    axis.set_facecolor("#10141f")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#324866")
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color="#c8cfdb",
        fontsize=12,
    )
    canvas.draw_idle()


def dark_plot_context():
    """Возвращает контекст dark-theme параметров matplotlib."""
    return mpl.rc_context(DARK_MATPLOTLIB_RC)

