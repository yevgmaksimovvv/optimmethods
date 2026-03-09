import logging
import math
import time
from typing import Iterable, List, Optional, Tuple

from matplotlib.figure import Figure

if __package__:
    from .app_models import FunctionSpec, SearchResult
    from .logging_setup import configure_logging
else:
    from app_models import FunctionSpec, SearchResult
    from logging_setup import configure_logging


configure_logging()
logger = logging.getLogger("lr1.plotting")


def _sample_function(
    function_spec: FunctionSpec,
    plot_range: Tuple[float, float],
    samples: int = 1600,
) -> Tuple[List[float], List[float]]:
    left, right = plot_range
    xs: List[float] = []
    ys: List[float] = []

    for index in range(samples + 1):
        x = left + (right - left) * index / samples
        if any(abs(x - point) < 1e-4 for point in function_spec.forbidden_points):
            xs.append(x)
            ys.append(float("nan"))
            continue
        try:
            y = function_spec.func(x)
        except ZeroDivisionError:
            xs.append(x)
            ys.append(float("nan"))
            continue

        xs.append(x)
        ys.append(y if math.isfinite(y) else float("nan"))

    return xs, ys


def _plot_single_method(
    axis,
    function_spec: FunctionSpec,
    result: SearchResult,
    xs: List[float],
    ys: List[float],
    color: str,
    reference_x: Optional[float],
    reference_f: Optional[float],
) -> None:
    axis.plot(xs, ys, color="#7ec8ff", linewidth=2.0, label=function_spec.title)

    a0, b0 = result.interval_initial
    af, bf = result.interval_final
    axis.axvline(a0, color="#7f8aa3", linestyle="--", alpha=0.45, label="Исходный интервал")
    axis.axvline(b0, color="#7f8aa3", linestyle="--", alpha=0.45)
    axis.axvline(af, color=color, linestyle=":", alpha=0.8, label="Финальный интервал")
    axis.axvline(bf, color=color, linestyle=":", alpha=0.8)

    if result.iterations:
        steps = list(range(1, len(result.iterations) + 1))
        lam_x = [row.lam for row in result.iterations]
        lam_y = [row.f_lam for row in result.iterations]
        mu_x = [row.mu for row in result.iterations]
        mu_y = [row.f_mu for row in result.iterations]

        axis.scatter(
            lam_x,
            lam_y,
            c=steps,
            cmap="Blues",
            s=28,
            alpha=0.9,
            marker="o",
            label="Шаги λ(k)",
            zorder=4,
        )
        axis.scatter(
            mu_x,
            mu_y,
            c=steps,
            cmap="Oranges",
            s=28,
            alpha=0.9,
            marker="s",
            label="Шаги μ(k)",
            zorder=4,
        )
        axis.plot(lam_x, lam_y, color="#6aa9ff", alpha=0.22, linewidth=1.0)
        axis.plot(mu_x, mu_y, color="#ffb36a", alpha=0.22, linewidth=1.0)

    if reference_x is not None and reference_f is not None and math.isfinite(reference_f):
        axis.scatter(
            [reference_x],
            [reference_f],
            marker="*",
            s=180,
            color="#ffd166",
            edgecolors="#806000",
            linewidths=0.6,
            label="Теоретический ориентир",
            zorder=5,
        )

    axis.scatter(
        [result.x_opt],
        [result.f_opt],
        s=90,
        color=color,
        edgecolors="#101317",
        linewidths=0.8,
        label=f"Найденное решение x*={result.x_opt:.5f}",
        zorder=6,
    )

    for point in function_spec.forbidden_points:
        if xs[0] < point < xs[-1]:
            axis.axvline(point, color="#d16b86", linestyle="-.", alpha=0.85, label=f"Разрыв x={point:g}")

    axis.set_title(f"{result.method} | итераций: {len(result.iterations)} | вызовов: {result.func_evals}")
    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.grid(True, alpha=0.25)

    handles, labels = axis.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    axis.legend(unique_handles, unique_labels, fontsize=8, loc="best")


def build_plot_figure(
    function_spec: FunctionSpec,
    results: Iterable[SearchResult],
    plot_range: Tuple[float, float],
    reference_x: Optional[float] = None,
    reference_f: Optional[float] = None,
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

    xs, ys = _sample_function(function_spec, plot_range)
    subplot_count = max(1, len(results))
    figure = Figure(figsize=(8.8, 3.8 * subplot_count), dpi=100)
    colors = ("#4ea8de", "#f77f00", "#55a630", "#c77dff")

    if results:
        axes = figure.subplots(subplot_count, 1, squeeze=False)
        for index, result in enumerate(results):
            axis = axes[index][0]
            _plot_single_method(
                axis=axis,
                function_spec=function_spec,
                result=result,
                xs=xs,
                ys=ys,
                color=colors[index % len(colors)],
                reference_x=reference_x,
                reference_f=reference_f,
            )
    else:
        axis = figure.add_subplot(111)
        axis.plot(xs, ys, color="#7ec8ff", linewidth=2.0, label=function_spec.title)
        axis.set_title(function_spec.title)
        axis.set_xlabel("x")
        axis.set_ylabel("f(x)")
        axis.grid(True, alpha=0.25)

    figure.tight_layout()
    logger.info(
        "build_plot_figure done sampled_points=%d duration_ms=%.2f",
        len(xs),
        (time.perf_counter() - started) * 1000.0,
    )
    return figure
