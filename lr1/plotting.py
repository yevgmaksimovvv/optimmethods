import logging
import math
import time
from typing import Iterable, Tuple

from matplotlib.figure import Figure

if __package__:
    from .app_models import FunctionSpec, SearchResult
    from .logging_setup import configure_logging
else:
    from app_models import FunctionSpec, SearchResult
    from logging_setup import configure_logging


configure_logging()
logger = logging.getLogger("lr1.plotting")


def build_plot_figure(
    function_spec: FunctionSpec,
    results: Iterable[SearchResult],
    plot_range: Tuple[float, float],
) -> Figure:
    started = time.perf_counter()
    results = list(results)
    logger.info(
        "build_plot_figure start function=%s results=%d plot_range=%s forbidden_points=%s",
        function_spec.key,
        len(results),
        plot_range,
        function_spec.forbidden_points,
    )
    left, right = plot_range
    xs = []
    ys = []
    n = 1600

    for index in range(n + 1):
        x = left + (right - left) * index / n
        if any(abs(x - point) < 1e-4 for point in function_spec.forbidden_points):
            continue
        try:
            y = function_spec.func(x)
        except ZeroDivisionError:
            continue
        if math.isfinite(y):
            xs.append(x)
            ys.append(y)

    figure = Figure(figsize=(8, 4.8), dpi=100)
    axis = figure.add_subplot(111)
    axis.plot(xs, ys, label=function_spec.title)

    for result in results:
        a0, b0 = result.interval_initial
        af, bf = result.interval_final
        axis.axvline(a0, linestyle="--", alpha=0.45, label=f"Init [{a0:.3f}, {b0:.3f}]")
        axis.axvline(b0, linestyle="--", alpha=0.45)
        axis.axvline(af, linestyle=":", alpha=0.75, label=f"Final [{af:.5f}, {bf:.5f}]")
        axis.axvline(bf, linestyle=":", alpha=0.75)
        axis.scatter([result.x_opt], [result.f_opt], s=45, label=f"x* = {result.x_opt:.5f} ({result.method})")

    for point in function_spec.forbidden_points:
        if left < point < right:
            axis.axvline(point, linestyle="-.", alpha=0.8, label=f"Discontinuity x={point:g}")

    axis.set_title(function_spec.title)
    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    logger.info(
        "build_plot_figure done sampled_points=%d duration_ms=%.2f",
        len(xs),
        (time.perf_counter() - started) * 1000.0,
    )
    return figure
