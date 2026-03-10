import logging
import math
import time
from typing import Iterable, List, Optional, Sequence, Tuple

from matplotlib import transforms
from matplotlib.figure import Figure

if __package__:
    from .app_models import FunctionSpec, GridRunResult, SearchResult
    from .logging_setup import configure_logging
else:
    from app_models import FunctionSpec, GridRunResult, SearchResult
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


def _focus_xlim(result: SearchResult, reference_x: Optional[float]) -> Tuple[float, float]:
    focus_points = [
        result.interval_initial[0],
        result.interval_initial[1],
        result.interval_final[0],
        result.interval_final[1],
        result.x_opt,
    ]
    if reference_x is not None:
        focus_points.append(reference_x)

    left = min(focus_points)
    right = max(focus_points)
    span = right - left
    if span <= 1e-9:
        span = max(result.interval_initial[1] - result.interval_initial[0], 1.0)
    half_span = (span / 2.0) + max(span * 0.08, (result.interval_final[1] - result.interval_final[0]) * 6.0, 0.25)
    center = (left + right) / 2.0
    return center - half_span, center + half_span


def _trimmed_bounds(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return None
    if len(finite_values) < 20:
        return finite_values[0], finite_values[-1]

    last_index = len(finite_values) - 1
    low_index = int(last_index * 0.02)
    high_index = int(last_index * 0.98)
    return finite_values[low_index], finite_values[high_index]


def _focus_ylim(
    visible_ys: Sequence[float],
    result: SearchResult,
    reference_f: Optional[float],
) -> Tuple[float, float]:
    trimmed = _trimmed_bounds(visible_ys)
    points = [result.f_opt]
    points.extend(row.f_lam for row in result.iterations)
    points.extend(row.f_mu for row in result.iterations)
    if reference_f is not None and math.isfinite(reference_f):
        points.append(reference_f)

    finite_points = [value for value in points if math.isfinite(value)]
    if trimmed is None:
        if not finite_points:
            return -1.0, 1.0
        low = min(finite_points)
        high = max(finite_points)
    else:
        low, high = trimmed
        if finite_points:
            low = min(low, min(finite_points))
            high = max(high, max(finite_points))

    if abs(high - low) < 1e-9:
        padding = max(abs(high) * 0.15, 1.0)
    else:
        padding = max((high - low) * 0.14, 1.0)
    return low - padding, high + padding


def _draw_interval_bracket(
    axis,
    left: float,
    right: float,
    *,
    y_axes: float,
    color: str,
    label: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
) -> None:
    transform = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    axis.plot(
        [left, right],
        [y_axes, y_axes],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        transform=transform,
        label=label,
        zorder=2.6,
    )
    cap_low = max(0.0, y_axes - 0.025)
    cap_high = min(1.0, y_axes + 0.025)
    axis.plot(
        [left, left],
        [cap_low, cap_high],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        transform=transform,
        zorder=2.6,
    )
    axis.plot(
        [right, right],
        [cap_low, cap_high],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        transform=transform,
        zorder=2.6,
    )


def _plot_single_method(
    axis,
    function_spec: FunctionSpec,
    result: SearchResult,
    color: str,
    reference_x: Optional[float],
    reference_f: Optional[float],
) -> None:
    x_limits = _focus_xlim(result, reference_x)
    xs, ys = _sample_function(function_spec, x_limits, samples=1200)
    axis.plot(xs, ys, color="#54b9f2", linewidth=2.2, label=function_spec.title, zorder=2)

    a0, b0 = result.interval_initial
    af, bf = result.interval_final
    axis.axvspan(a0, b0, color="#8ea0bf", alpha=0.035, zorder=0)
    axis.axvspan(af, bf, color=color, alpha=0.08, zorder=1)
    _draw_interval_bracket(
        axis,
        a0,
        b0,
        y_axes=0.94,
        color="#9ba9bf",
        label="Исходный интервал",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
    )
    _draw_interval_bracket(
        axis,
        af,
        bf,
        y_axes=0.89,
        color=color,
        label="Финальный интервал",
        linestyle="-",
        linewidth=2.0,
        alpha=0.95,
    )

    if result.iterations:
        lam_x = [row.lam for row in result.iterations]
        lam_y = [row.f_lam for row in result.iterations]
        mu_x = [row.mu for row in result.iterations]
        mu_y = [row.f_mu for row in result.iterations]

        axis.plot(lam_x, lam_y, color="#0f6cbd", linewidth=1.7, alpha=1.0, zorder=3)
        axis.plot(mu_x, mu_y, color="#ff8c32", linewidth=1.7, alpha=1.0, zorder=3)
        axis.scatter(
            lam_x,
            lam_y,
            color="#0f6cbd",
            s=44,
            alpha=1.0,
            edgecolors="#dff3ff",
            linewidths=0.7,
            marker="o",
            label="Шаги λ(k)",
            zorder=4,
        )
        axis.scatter(
            mu_x,
            mu_y,
            color="#ff8c32",
            s=44,
            alpha=1.0,
            edgecolors="#fff1dd",
            linewidths=0.7,
            marker="s",
            label="Шаги μ(k)",
            zorder=4,
        )

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
            axis.axvline(point, color="#d16b86", linestyle="-.", alpha=0.95, linewidth=1.7, label=f"Разрыв x={point:g}")

    axis.set_title(f"{result.method} | итераций: {len(result.iterations)} | вызовов: {result.func_evals}")
    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*_focus_ylim(ys, result, reference_f))
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


def _grid_run_label(run: GridRunResult) -> str:
    return f"ε={run.eps:g}\nl={run.l:g}"


def _plot_grid_overview(
    axis,
    runs: Sequence[GridRunResult],
    selected_index: int,
    method_title: str,
) -> None:
    positions = list(range(len(runs)))
    eval_values = [run.result.func_evals for run in runs]
    x_opt_values = [run.result.x_opt for run in runs]
    labels = [_grid_run_label(run) for run in runs]

    colors = ["#33415c"] * len(runs)
    if 0 <= selected_index < len(runs):
        colors[selected_index] = "#2f81f7"

    bars = axis.bar(positions, eval_values, color=colors, edgecolor="#7f8aa3", linewidth=0.8)
    axis.set_title(f"Серия запусков: {method_title}")
    axis.set_ylabel("Вызовы функции")
    axis.set_xticks(positions, labels)
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_axisbelow(True)

    max_eval = max(eval_values) if eval_values else 0
    offset = max(0.35, max_eval * 0.03)
    for index, (bar, eval_value) in enumerate(zip(bars, eval_values)):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            str(eval_value),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#eef2f8" if index == selected_index else "#c8cfdb",
        )

    axis_right = axis.twinx()
    axis_right.plot(
        positions,
        x_opt_values,
        color="#ffd166",
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="x*",
    )
    if 0 <= selected_index < len(runs):
        axis_right.scatter(
            [selected_index],
            [x_opt_values[selected_index]],
            color="#ff9f1c",
            s=80,
            zorder=5,
            label="Выбранный прогон",
        )
    axis_right.set_ylabel("x*")

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = axis_right.get_legend_handles_labels()
    if handles_right or handles_left:
        axis.legend(handles_left + handles_right, labels_left + labels_right, fontsize=8, loc="upper left")


def build_grid_plot_figure(
    function_spec: FunctionSpec,
    runs: Sequence[GridRunResult],
    selected_index: int,
    plot_range: Tuple[float, float],
    method_title: str,
    reference_x: Optional[float] = None,
    reference_f: Optional[float] = None,
) -> Figure:
    started = time.perf_counter()
    sampled_points = 0
    logger.info(
        "build_grid_plot_figure start function=%s runs=%d selected_index=%d plot_range=%s",
        function_spec.key,
        len(runs),
        selected_index,
        plot_range,
    )

    figure = Figure(figsize=(10.4, 8.2), dpi=100)
    top_axis, bottom_axis = figure.subplots(2, 1, gridspec_kw={"height_ratios": [1.15, 2.4]})

    if runs:
        selected_index = min(max(selected_index, 0), len(runs) - 1)
        _plot_grid_overview(top_axis, runs, selected_index, method_title)
        selected_run = runs[selected_index]
        _plot_single_method(
            axis=bottom_axis,
            function_spec=function_spec,
            result=selected_run.result,
            color="#2f81f7",
            reference_x=reference_x,
            reference_f=reference_f,
        )
        bottom_axis.set_title(
            f"{selected_run.result.method} | ε={selected_run.eps:g}, l={selected_run.l:g} | "
            f"итераций: {len(selected_run.result.iterations)} | вызовов: {selected_run.result.func_evals}"
        )
    else:
        xs, ys = _sample_function(function_spec, plot_range)
        sampled_points = len(xs)
        top_axis.set_title(f"Серия запусков: {method_title}")
        top_axis.text(
            0.5,
            0.5,
            "Нет данных для построения серии.",
            ha="center",
            va="center",
            transform=top_axis.transAxes,
            color="#c8cfdb",
        )
        top_axis.set_axis_off()
        bottom_axis.plot(xs, ys, color="#7ec8ff", linewidth=2.0, label=function_spec.title)
        bottom_axis.set_title(function_spec.title)
        bottom_axis.set_xlabel("x")
        bottom_axis.set_ylabel("f(x)")
        bottom_axis.grid(True, alpha=0.25)

    figure.tight_layout()
    logger.info(
        "build_grid_plot_figure done sampled_points=%d duration_ms=%.2f",
        sampled_points,
        (time.perf_counter() - started) * 1000.0,
    )
    return figure


def build_plot_figure(
    function_spec: FunctionSpec,
    results: Iterable[SearchResult],
    plot_range: Tuple[float, float],
    reference_x: Optional[float] = None,
    reference_f: Optional[float] = None,
) -> Figure:
    started = time.perf_counter()
    results = list(results)
    sampled_points = 0
    logger.info(
        "build_plot_figure start function=%s results=%d plot_range=%s forbidden_points=%s",
        function_spec.key,
        len(results),
        plot_range,
        function_spec.forbidden_points,
    )

    subplot_count = max(1, len(results))
    figure = Figure(figsize=(10.4, 3.9 * subplot_count), dpi=100)
    colors = ("#4ea8de", "#f77f00", "#55a630", "#c77dff")

    if results:
        axes = figure.subplots(subplot_count, 1, squeeze=False)
        for index, result in enumerate(results):
            axis = axes[index][0]
            _plot_single_method(
                axis=axis,
                function_spec=function_spec,
                result=result,
                color=colors[index % len(colors)],
                reference_x=reference_x,
                reference_f=reference_f,
            )
    else:
        xs, ys = _sample_function(function_spec, plot_range)
        sampled_points = len(xs)
        axis = figure.add_subplot(111)
        axis.plot(xs, ys, color="#7ec8ff", linewidth=2.0, label=function_spec.title)
        axis.set_title(function_spec.title)
        axis.set_xlabel("x")
        axis.set_ylabel("f(x)")
        axis.grid(True, alpha=0.25)

    figure.tight_layout()
    logger.info(
        "build_plot_figure done sampled_points=%d duration_ms=%.2f",
        sampled_points,
        (time.perf_counter() - started) * 1000.0,
    )
    return figure
