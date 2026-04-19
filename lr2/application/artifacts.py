"""Сохранение артефактов и вычисление сетки для ЛР2."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401  # регистрация 3d-проекции

from lr2.domain.models import BATCH_STATUS_SUCCESS, BatchResult, BatchItemResult, SolverResult
from lr2.domain.polynomial import evaluate_polynomial, format_polynomial
from optim_core.ui import dark_plot_context

ARTIFACTS_BASE_DIR = Path("report") / "lr2_runs"


class RosenbrockArtifactsStore:
    """Сохраняет batch-артефакты ЛР2 и строит данные для графиков."""

    def __init__(self, base_dir: Path = ARTIFACTS_BASE_DIR) -> None:
        self._base_dir = base_dir

    def save_batch_result(self, batch_result: BatchResult, trace_id: str) -> Path:
        run_dir = self._create_artifacts_dir(trace_id)
        formula_text = format_polynomial(batch_result.polynomial)
        self._write_summary_csv(run_dir / "summary.csv", batch_result)
        (run_dir / "formula.txt").write_text(f"Формула: {formula_text}\n", encoding="utf-8")

        run_number = 0
        for item in batch_result.items or tuple(
            BatchItemResult(
                epsilon=run.epsilon,
                start_point=run.start_point,
                status=BATCH_STATUS_SUCCESS,
                run=run,
            )
            for run in batch_result.runs
        ):
            if item.run is None:
                continue
            run_number += 1
            single_dir = run_dir / f"run_{run_number:03d}"
            single_dir.mkdir(parents=True, exist_ok=True)
            self._write_iterations_csv(single_dir / "iterations.csv", item.run)
            self._save_run_plot_png(batch_result, item.run, mode="contour", output_path=single_dir / "contour.png")
            self._save_run_plot_png(batch_result, item.run, mode="surface", output_path=single_dir / "surface.png")

        return run_dir

    def build_mesh(self, batch_result: BatchResult, mesh_x: np.ndarray, mesh_y: np.ndarray) -> np.ndarray:
        polynomial = batch_result.polynomial
        result = np.zeros_like(mesh_x, dtype=float)
        for i, row in enumerate(polynomial.coefficients):
            x_part = np.power(mesh_x, i)
            for j, coefficient in enumerate(row):
                if coefficient == 0.0:
                    continue
                result += coefficient * x_part * np.power(mesh_y, j)
        return result

    def _create_artifacts_dir(self, trace_id: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{trace_id}"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        candidate = self._base_dir / base_name
        suffix = 1
        while candidate.exists():
            candidate = self._base_dir / f"{base_name}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _write_summary_csv(self, path: Path, batch_result: BatchResult) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([
                "#",
                "epsilon",
                "start",
                "x*",
                "f(x*)",
                "N",
                "status",
                "detail",
                "exception",
            ])
            for index, item in enumerate(batch_result.items or self._items_from_runs(batch_result), start=1):
                if item.run is None:
                    writer.writerow(
                        [
                            index,
                            f"{item.epsilon:.6g}",
                            self._format_point(item.start_point),
                            "",
                            "",
                            "",
                            item.status,
                            item.message or "",
                            item.exception_type or "",
                        ]
                    )
                    continue
                run = item.run
                writer.writerow(
                    [
                        index,
                        f"{run.epsilon:.6g}",
                        self._format_point(run.start_point),
                        self._format_point(run.optimum_point),
                        f"{run.optimum_value:.8g}",
                        run.iterations_count,
                        item.status,
                        run.stop_reason,
                        "",
                    ]
                )

    def _write_iterations_csv(self, path: Path, run: SolverResult) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["K", "x_k", "F(x_k)", "j", "d_j", "y_j", "f(y_j)", "λ_j", "y_{j+1}", "f(y_{j+1})"])
            for step in run.steps:
                writer.writerow(
                    [
                        step.k,
                        self._format_point(step.x_k),
                        f"{step.f_x_k:.8g}",
                        step.j,
                        self._format_point(step.direction),
                        self._format_point(step.y_j),
                        f"{step.f_y_j:.8g}",
                        f"{step.lambda_j:.8g}",
                        self._format_point(step.y_next),
                        f"{step.f_y_next:.8g}",
                    ]
                )

    def _save_run_plot_png(self, batch_result: BatchResult, run: SolverResult, mode: str, output_path: Path) -> None:
        points = np.array(run.trajectory)
        if points.ndim != 2 or points.shape[1] != 2:
            return

        x_vals = points[:, 0]
        y_vals = points[:, 1]
        x_min = float(np.min(x_vals))
        x_max = float(np.max(x_vals))
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        span_x = max(x_max - x_min, 1.0)
        span_y = max(y_max - y_min, 1.0)
        margin_x = span_x * 0.6
        margin_y = span_y * 0.6

        grid_x = np.linspace(x_min - margin_x, x_max + margin_x, 120)
        grid_y = np.linspace(y_min - margin_y, y_max + margin_y, 120)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        mesh_z = self.build_mesh(batch_result, mesh_x, mesh_y)

        with dark_plot_context():
            figure = Figure(figsize=(9.0, 6.0), dpi=120)
            try:
                figure.patch.set_facecolor("#171b24")
                if mode == "surface":
                    ax_surface = figure.add_subplot(1, 1, 1, projection="3d")
                    if hasattr(ax_surface, "set_proj_type"):
                        ax_surface.set_proj_type("ortho")
                    z_clipped = self._build_surface_mesh(mesh_z)
                    ax_surface.plot_surface(
                        mesh_x,
                        mesh_y,
                        z_clipped,
                        cmap="turbo",
                        alpha=0.68,
                        linewidth=0,
                        antialiased=True,
                    )
                    path_z = [evaluate_polynomial(batch_result.polynomial, point[0], point[1]) for point in run.trajectory]
                    z_span = max(float(np.nanmax(z_clipped) - np.nanmin(z_clipped)), 1.0)
                    lifted_path_z = [value + z_span * 0.08 for value in path_z]
                    self._draw_surface_trajectory(ax_surface, x_vals, y_vals, lifted_path_z)
                    ax_surface.set_title("Поверхность и траектория")
                    ax_surface.set_xlabel("x1")
                    ax_surface.set_ylabel("x2")
                    ax_surface.set_zlabel("f(x1, x2)")
                    ax_surface.grid(False)
                    spans = self._report_surface_aspect(x_max - x_min, y_max - y_min, z_span)
                    if hasattr(ax_surface, "set_box_aspect"):
                        ax_surface.set_box_aspect(spans)
                    for axis in (ax_surface.xaxis, ax_surface.yaxis, ax_surface.zaxis):
                        pane = getattr(axis, "pane", None)
                        if pane is not None:
                            pane.set_facecolor((0.12, 0.16, 0.23, 0.45))
                    ax_surface.tick_params(colors="#c4cfdf")
                    surface_top = float(np.nanmax(z_clipped))
                    ax_surface.set_zlim(float(np.nanmin(z_clipped)), float(surface_top + z_span * 0.2))
                    elev, azim = self._surface_view_angles(points)
                    ax_surface.view_init(elev=elev, azim=azim)
                else:
                    ax_contour = figure.add_subplot(1, 1, 1)
                    contour = ax_contour.contour(mesh_x, mesh_y, mesh_z, levels=24, cmap="turbo")
                    ax_contour.set_facecolor("#10141f")
                    ax_contour.clabel(contour, inline=True, fontsize=8, colors="#dce6f5")
                    ax_contour.plot(
                        x_vals,
                        y_vals,
                        marker="o",
                        color="#ffffff",
                        linewidth=3.1,
                        markersize=4.5,
                        markerfacecolor="#ff4f87",
                        markeredgewidth=0.0,
                        zorder=3,
                    )
                    ax_contour.scatter([x_vals[0]], [y_vals[0]], color="#2da3ff", s=100, label="Старт", zorder=4)
                    ax_contour.scatter([x_vals[-1]], [y_vals[-1]], color="#57d773", s=100, label="Финиш", zorder=4)
                    ax_contour.set_title("Линии уровня + траектория")
                    ax_contour.set_xlabel("x1")
                    ax_contour.set_ylabel("x2")
                    ax_contour.set_aspect("equal", adjustable="box")
                    ax_contour.grid(True)
                    legend = ax_contour.legend(loc="upper right", framealpha=0.92)
                    for text in legend.get_texts():
                        text.set_color("#e8f0ff")

                figure.savefig(output_path, dpi=120, bbox_inches="tight")
            finally:
                figure.clf()

    @staticmethod
    def _items_from_runs(batch_result: BatchResult) -> tuple[BatchItemResult, ...]:
        return tuple(
            BatchItemResult(
                epsilon=run.epsilon,
                start_point=run.start_point,
                status=BATCH_STATUS_SUCCESS,
                run=run,
            )
            for run in batch_result.runs
        )

    @staticmethod
    def _surface_view_angles(points: np.ndarray) -> tuple[float, float]:
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            return 30.0, -55.0

        xy = np.asarray(points[:, :2], dtype=float)
        centered = xy - np.mean(xy, axis=0, keepdims=True)
        if not np.isfinite(centered).all():
            return 30.0, -55.0

        cov = np.cov(centered, rowvar=False)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            return 30.0, -55.0

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        azim = float(np.degrees(np.arctan2(principal[1], principal[0]))) + 90.0
        if azim > 180.0:
            azim -= 360.0
        if azim <= -180.0:
            azim += 360.0

        spans = np.ptp(xy, axis=0)
        major = float(max(spans.max(), 1.0))
        minor = float(max(spans.min(), 1e-6))
        aspect = major / minor
        elev = 30.0 + min(10.0, max(0.0, math.log1p(max(aspect - 1.0, 0.0)) * 4.0))
        return elev, azim

    @staticmethod
    def _draw_surface_trajectory(
        ax_surface: _Axes3D,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        lifted_path_z: list[float],
    ) -> None:
        ax_surface.plot(
            x_vals,
            y_vals,
            lifted_path_z,
            color="#0f131a",
            linewidth=12.0,
            marker="o",
            markersize=6.6,
            markerfacecolor="#0f131a",
            markeredgewidth=0.0,
            alpha=1.0,
        )
        ax_surface.plot(
            x_vals,
            y_vals,
            lifted_path_z,
            color="#ff2d95",
            linewidth=6.8,
            marker="o",
            markersize=5.4,
            markerfacecolor="#ff4f87",
            markeredgecolor="#ffffff",
            markeredgewidth=1.0,
            alpha=1.0,
        )
        ax_surface.plot(
            x_vals,
            y_vals,
            lifted_path_z,
            color="#ffffff",
            linewidth=3.4,
            marker="o",
            markersize=4.6,
            markerfacecolor="#ff4f87",
            markeredgecolor="#ffffff",
            markeredgewidth=0.8,
            alpha=1.0,
        )
        ax_surface.scatter(
            [x_vals[0]],
            [y_vals[0]],
            [lifted_path_z[0]],
            color="#2da3ff",
            edgecolors="#ffffff",
            linewidths=1.0,
            s=160,
            depthshade=False,
            alpha=1.0,
        )
        ax_surface.scatter(
            [x_vals[-1]],
            [y_vals[-1]],
            [lifted_path_z[-1]],
            color="#57d773",
            edgecolors="#ffffff",
            linewidths=1.0,
            s=160,
            depthshade=False,
            alpha=1.0,
        )

    @staticmethod
    def _report_surface_aspect(x_span: float, y_span: float, z_span: float) -> tuple[float, float, float]:
        xy_span = max(x_span, y_span, 1.0)
        z_display_span = min(max(z_span * 0.35, xy_span * 0.35), xy_span * 1.25)
        return xy_span, xy_span, z_display_span

    @staticmethod
    def _build_surface_mesh(mesh_z: np.ndarray) -> np.ndarray:
        finite_values = mesh_z[np.isfinite(mesh_z)]
        if finite_values.size == 0:
            return np.zeros_like(mesh_z, dtype=float)
        low = float(np.percentile(finite_values, 5))
        high = float(np.percentile(finite_values, 95))
        if high < low:
            low, high = high, low
        clipped = np.clip(np.nan_to_num(mesh_z, nan=0.0, posinf=high, neginf=low), low, high)
        return clipped

    @staticmethod
    def _format_point(point: tuple[float, ...]) -> str:
        return "(" + ", ".join(f"{value:.6g}" for value in point) + ")"
