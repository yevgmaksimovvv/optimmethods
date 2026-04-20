"""Сохранение отчётных артефактов ЛР5."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lr2.domain.models import SolverResult, Vector
from lr5.domain.models import BarrierResult
from optim_core.ui import dark_plot_context

ARTIFACTS_BASE_DIR = Path("report") / "lr5_runs"


class BarrierArtifactsStore:
    """Сохраняет таблицы и графики для ЛР5."""

    def __init__(self, base_dir: Path = ARTIFACTS_BASE_DIR) -> None:
        self._base_dir = base_dir

    def save_result(self, result: BarrierResult, trace_id: str) -> Path:
        run_dir = self._create_artifacts_dir(trace_id)
        self._write_summary_csv(run_dir / "summary.csv", result)
        self._write_summary_txt(run_dir / "summary.txt", result)
        self._save_main_plot(run_dir / "contour.png", result)
        for iteration in result.iterations:
            iteration_dir = run_dir / f"outer_{iteration.k:02d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            self._write_inner_iterations_csv(iteration_dir / "inner_iterations.csv", iteration.inner_result)
            self._write_trajectory_csv(iteration_dir / "trajectory.csv", iteration.inner_result)
        return run_dir

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

    def _write_summary_csv(self, path: Path, result: BarrierResult) -> None:
        constraint_headers = [f"g{index + 1}(x)" for index in range(len(result.problem.constraints))]
        with path.open("w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(
                [
                    "k",
                    "mu_k",
                    "x_{mu_k}",
                    "F(x_{mu_k})",
                    "M(x_{mu_k})",
                    "mu_k * M(x_{mu_k})",
                    "Theta(mu_k)",
                    *constraint_headers,
                    "inner_iterations",
                    "inner_success",
                    "inner_stop_reason",
                ]
            )
            for iteration in result.iterations:
                writer.writerow(
                    [
                        iteration.k,
                        f"{iteration.mu_k:.8g}",
                        self._format_point(iteration.x_mu_k),
                        f"{iteration.objective_value:.10g}",
                        f"{iteration.barrier_metric:.10g}",
                        f"{iteration.barrier_metric_term:.10g}",
                        f"{iteration.theta_value:.10g}",
                        *[f"{value:.10g}" for value in iteration.constraints_values],
                        iteration.inner_result.iterations_count,
                        iteration.inner_result.success,
                        iteration.inner_result.stop_reason,
                    ]
                )

    def _write_summary_txt(self, path: Path, result: BarrierResult) -> None:
        last_valid = result.last_valid_outer_iteration
        if last_valid is not None:
            last_valid_point = self._format_point(last_valid.x_mu_k)
            last_valid_value = f"{last_valid.objective_value:.10g}"
            last_valid_constraints = ", ".join(f"{value:.10g}" for value in last_valid.constraints_values)
            last_valid_mu = f"{last_valid.mu_k:.10g}"
        else:
            last_valid_point = "—"
            last_valid_value = "—"
            last_valid_constraints = "—"
            last_valid_mu = "—"
        path.write_text(
            "\n".join(
                [
                    f"Точка старта: {self._format_point(result.start_point)}",
                    f"Статус: {result.status}",
                    f"Последняя допустимая точка: {last_valid_point}",
                    f"F(x_last_valid): {last_valid_value}",
                    f"g(x_last_valid): {last_valid_constraints}",
                    f"mu_last_valid: {last_valid_mu}",
                    "Критерий внешнего цикла: mu_k * M(x_mu_k) < epsilon_outer.",
                    "M(x)=Σ 1/(-g_i(x)); для log barrier внутренняя цель остаётся "
                    "-Σ ln(-g_i(x)), а внешний стоп использует ту же общую метрику.",
                    f"success: {result.success}",
                    f"last_valid_present: {last_valid is not None}",
                    f"stop_reason: {result.stop_reason}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_inner_iterations_csv(self, path: Path, run: SolverResult) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["K", "x_k", "F(x_k)", "j", "d_j", "y_j", "f(y_j)", "lambda_j", "y_{j+1}", "f(y_{j+1})"])
            for step in run.steps:
                writer.writerow(
                    [
                        step.k,
                        self._format_point(step.x_k),
                        f"{step.f_x_k:.10g}",
                        step.j,
                        self._format_point(step.direction),
                        self._format_point(step.y_j),
                        f"{step.f_y_j:.10g}",
                        f"{step.lambda_j:.10g}",
                        self._format_point(step.y_next),
                        f"{step.f_y_next:.10g}",
                    ]
                )

    def _write_trajectory_csv(self, path: Path, run: SolverResult) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["#", "x"])
            for index, point in enumerate(run.trajectory, start=1):
                writer.writerow([index, self._format_point(point)])

    def _save_main_plot(self, output_path: Path, result: BarrierResult) -> None:
        points = self._collect_points(result)
        x_min, x_max, y_min, y_max = self._plot_bounds(points)
        grid_x = np.linspace(x_min, x_max, 320)
        grid_y = np.linspace(y_min, y_max, 320)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        objective_mesh = self._build_objective_mesh(result, mesh_x, mesh_y)
        g1_mesh = -mesh_x + 2.0 * mesh_y - 4.0
        g2_mesh = mesh_x + mesh_y - 3.0
        feasible_mask = np.where((g1_mesh <= 0.0) & (g2_mesh <= 0.0), 1.0, 0.0)

        with dark_plot_context():
            figure = Figure(figsize=(10.8, 7.2), dpi=120)
            ax = figure.add_subplot(1, 1, 1)
            figure.patch.set_facecolor("#171b24")
            ax.set_facecolor("#10141f")
            ax.contourf(mesh_x, mesh_y, feasible_mask, levels=[-0.5, 0.5, 1.5], colors=["#1f7a3d"], alpha=0.18)
            contour = ax.contour(mesh_x, mesh_y, objective_mesh, levels=22, cmap="turbo", linewidths=1.0)
            ax.clabel(contour, inline=True, fontsize=8, colors="#dce6f5")
            ax.contour(mesh_x, mesh_y, g1_mesh, levels=[0.0], colors="#ff8c42", linewidths=2.4)
            ax.contour(mesh_x, mesh_y, g2_mesh, levels=[0.0], colors="#6ee7ff", linewidths=2.4)

            outer_points = [result.start_point, *[iteration.x_mu_k for iteration in result.iterations]]
            if outer_points:
                outer_x = [point[0] for point in outer_points]
                outer_y = [point[1] for point in outer_points]
                ax.plot(outer_x, outer_y, color="#ffffff", linewidth=3.0, marker="o", markersize=5.0, zorder=4)
                ax.scatter([outer_x[0]], [outer_y[0]], color="#2da3ff", s=110, label="Старт", zorder=5)
                if result.last_valid_outer_iteration is not None:
                    final_label = "Финиш" if result.status == "success" else "Последняя допустимая точка"
                    ax.scatter([outer_x[-1]], [outer_y[-1]], color="#57d773", s=110, label=final_label, zorder=5)

            for iteration in result.iterations:
                trajectory = list(iteration.inner_result.trajectory)
                if len(trajectory) < 2:
                    continue
                xs = [point[0] for point in trajectory]
                ys = [point[1] for point in trajectory]
                ax.plot(xs, ys, color="#ff4f87", alpha=0.6, linewidth=1.8, marker="o", markersize=3.8, zorder=3)

            ax.set_title("Метод барьерных функций: линии уровня, ограничения и траектории")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.grid(alpha=0.18)
            ax.legend(loc="best")
            figure.tight_layout()
            figure.savefig(output_path, facecolor=figure.get_facecolor())

    def _collect_points(self, result: BarrierResult) -> list[Vector]:
        points: list[Vector] = [result.start_point, result.optimum_point]
        for iteration in result.iterations:
            points.extend(iteration.inner_result.trajectory)
            points.append(iteration.x_mu_k)
        return points

    def _plot_bounds(self, points: list[Vector]) -> tuple[float, float, float, float]:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        margin_x = x_span * 0.75
        margin_y = y_span * 0.75
        return x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y

    def _build_objective_mesh(self, result: BarrierResult, mesh_x: np.ndarray, mesh_y: np.ndarray) -> np.ndarray:
        objective = result.problem.objective
        mesh = np.zeros_like(mesh_x, dtype=float)
        for index in np.ndindex(mesh_x.shape):
            mesh[index] = objective((float(mesh_x[index]), float(mesh_y[index])))
        return mesh

    @staticmethod
    def _format_point(point: Vector) -> str:
        return f"({point[0]:.10g}; {point[1]:.10g})"
