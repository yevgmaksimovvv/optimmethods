"""Tkinter GUI для ЛР3."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lr3.application.services import (
    DEFAULT_CONJUGATE_EXPRESSION,
    DEFAULT_GRADIENT_EXPRESSION,
    build_config,
    build_start_point,
    run_conjugate,
    run_gradient,
)
from lr3.domain.expression import compile_objective
from lr3.domain.models import OptimizationResult

APP_TITLE = "ЛР3 — градиентные методы"


class Lr3Window:
    """Главное окно ЛР3."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1440x860")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_method_tab(
            notebook=notebook,
            tab_title="Градиентный метод",
            default_expression=DEFAULT_GRADIENT_EXPRESSION,
            default_x1="0",
            default_x2="0",
            default_step="0.1",
            default_epsilon="1e-6",
            default_iterations="1000",
            default_timeout="3.0",
            run_callback=self._run_gradient,
        )

        self._build_method_tab(
            notebook=notebook,
            tab_title="Сопряжённые градиенты",
            default_expression=DEFAULT_CONJUGATE_EXPRESSION,
            default_x1="1",
            default_x2="1",
            default_step="0.2",
            default_epsilon="1e-6",
            default_iterations="300",
            default_timeout="3.0",
            run_callback=self._run_conjugate,
        )

    def _build_method_tab(
        self,
        notebook: ttk.Notebook,
        tab_title: str,
        default_expression: str,
        default_x1: str,
        default_x2: str,
        default_step: str,
        default_epsilon: str,
        default_iterations: str,
        default_timeout: str,
        run_callback,
    ) -> dict[str, object]:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=tab_title)

        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        right = ttk.Frame(frame)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        form = ttk.LabelFrame(left, text="Параметры", padding=10)
        form.pack(fill="x", pady=6)

        ttk.Label(form, text="F(x1, x2) =").grid(row=0, column=0, sticky="w")
        expression_entry = ttk.Entry(form, width=52)
        expression_entry.grid(row=0, column=1, padx=6, pady=2, sticky="ew")
        expression_entry.insert(0, default_expression)

        ttk.Label(form, text="x1(0):").grid(row=1, column=0, sticky="w")
        x1_entry = ttk.Entry(form, width=18)
        x1_entry.grid(row=1, column=1, padx=6, pady=2, sticky="w")
        x1_entry.insert(0, default_x1)

        ttk.Label(form, text="x2(0):").grid(row=2, column=0, sticky="w")
        x2_entry = ttk.Entry(form, width=18)
        x2_entry.grid(row=2, column=1, padx=6, pady=2, sticky="w")
        x2_entry.insert(0, default_x2)

        ttk.Label(form, text="Начальный шаг:").grid(row=3, column=0, sticky="w")
        step_entry = ttk.Entry(form, width=18)
        step_entry.grid(row=3, column=1, padx=6, pady=2, sticky="w")
        step_entry.insert(0, default_step)

        ttk.Label(form, text="epsilon:").grid(row=4, column=0, sticky="w")
        epsilon_entry = ttk.Entry(form, width=18)
        epsilon_entry.grid(row=4, column=1, padx=6, pady=2, sticky="w")
        epsilon_entry.insert(0, default_epsilon)

        ttk.Label(form, text="max_iterations:").grid(row=5, column=0, sticky="w")
        iterations_entry = ttk.Entry(form, width=18)
        iterations_entry.grid(row=5, column=1, padx=6, pady=2, sticky="w")
        iterations_entry.insert(0, default_iterations)

        ttk.Label(form, text="timeout (сек):").grid(row=6, column=0, sticky="w")
        timeout_entry = ttk.Entry(form, width=18)
        timeout_entry.grid(row=6, column=1, padx=6, pady=2, sticky="w")
        timeout_entry.insert(0, default_timeout)

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=8)

        output = scrolledtext.ScrolledText(left, height=18)
        output.pack(fill="both", expand=True)

        run_button = ttk.Button(buttons, text="Запустить", command=lambda: run_callback(widgets))
        run_button.pack(side="left", padx=4)
        clear_button = ttk.Button(buttons, text="Очистить", command=lambda: output.delete("1.0", tk.END))
        clear_button.pack(side="left", padx=4)

        figure = plt.Figure(figsize=(7.0, 6.0), dpi=100)
        canvas = FigureCanvasTkAgg(figure, right)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        widgets: dict[str, object] = {
            "expression": expression_entry,
            "x1": x1_entry,
            "x2": x2_entry,
            "step": step_entry,
            "epsilon": epsilon_entry,
            "iterations": iterations_entry,
            "timeout": timeout_entry,
            "output": output,
            "figure": figure,
            "canvas": canvas,
            "run_button": run_button,
        }
        return widgets

    def _run_gradient(self, widgets: dict[str, object]) -> None:
        self._execute("gradient", widgets)

    def _run_conjugate(self, widgets: dict[str, object]) -> None:
        self._execute("conjugate", widgets)

    def _execute(self, method: str, widgets: dict[str, object]) -> None:
        try:
            expression = self._entry(widgets, "expression").get().strip()
            start_point = build_start_point(
                self._entry(widgets, "x1").get(),
                self._entry(widgets, "x2").get(),
            )
            config = build_config(
                epsilon_raw=self._entry(widgets, "epsilon").get(),
                max_iterations_raw=self._entry(widgets, "iterations").get(),
                initial_step_raw=self._entry(widgets, "step").get(),
                timeout_raw=self._entry(widgets, "timeout").get(),
            )

            if method == "gradient":
                result, metrics = run_gradient(expression, start_point, config)
            else:
                result, metrics = run_conjugate(expression, start_point, config)

            self._render_output(widgets, expression, result, metrics.trace_id, metrics.latency_ms)
            self._render_plot(widgets, expression, result)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _render_output(
        self,
        widgets: dict[str, object],
        expression: str,
        result: OptimizationResult,
        trace_id: str,
        latency_ms: float,
    ) -> None:
        output = self._output(widgets)
        output.delete("1.0", tk.END)

        lines = [
            f"Метод: {result.method_name}",
            f"Trace ID: {trace_id}",
            f"Функция: F(x1, x2) = {expression}",
            f"Старт: ({result.start_point[0]:.6f}, {result.start_point[1]:.6f})",
            f"Точка: ({result.optimum_point[0]:.8f}, {result.optimum_point[1]:.8f})",
            f"Значение: {result.optimum_value:.8f}",
            f"Итераций: {result.iterations_count}",
            f"Успех: {result.success}",
            f"Причина остановки: {result.stop_reason}",
            f"Время: {latency_ms:.2f} ms",
            "",
            f"{'k':>4} | {'x1':>12} | {'x2':>12} | {'F(x)':>12} | {'||grad||':>12} | {'step':>10}",
            "-" * 82,
        ]

        for record in result.records:
            grad_norm = (record.gradient[0] ** 2 + record.gradient[1] ** 2) ** 0.5
            lines.append(
                f"{record.k:>4} | {record.point[0]:>12.6f} | {record.point[1]:>12.6f} | "
                f"{record.value:>12.6f} | {grad_norm:>12.4e} | {record.step_size:>10.6f}"
            )

        output.insert(tk.END, "\n".join(lines))

    def _render_plot(self, widgets: dict[str, object], expression: str, result: OptimizationResult) -> None:
        objective = compile_objective(expression)

        figure = self._figure(widgets)
        figure.clear()

        ax_contour = figure.add_subplot(121)
        ax_conv = figure.add_subplot(122)

        xs = [record.point[0] for record in result.records] or [result.start_point[0], result.optimum_point[0]]
        ys = [record.point[1] for record in result.records] or [result.start_point[1], result.optimum_point[1]]

        margin = 1.0
        x_min = min(xs) - margin
        x_max = max(xs) + margin
        y_min = min(ys) - margin
        y_max = max(ys) + margin

        grid_x = np.linspace(x_min, x_max, 80)
        grid_y = np.linspace(y_min, y_max, 80)
        x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)
        z_mesh = np.zeros_like(x_mesh)

        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                z_mesh[i, j] = objective((float(x_mesh[i, j]), float(y_mesh[i, j])))

        contour = ax_contour.contourf(x_mesh, y_mesh, z_mesh, levels=25, cmap="viridis")
        figure.colorbar(contour, ax=ax_contour)
        ax_contour.plot(xs, ys, "r.-", linewidth=2, markersize=6)
        ax_contour.scatter([xs[0]], [ys[0]], c="white", edgecolors="black", label="start")
        ax_contour.scatter([result.optimum_point[0]], [result.optimum_point[1]], c="red", label="optimum")
        ax_contour.set_title("Траектория на линии уровня")
        ax_contour.set_xlabel("x1")
        ax_contour.set_ylabel("x2")
        ax_contour.legend()

        values = [record.value for record in result.records]
        if not values:
            values = [result.optimum_value]
        ax_conv.plot(range(len(values)), values, "b-o", linewidth=2, markersize=4)
        ax_conv.set_title("Сходимость")
        ax_conv.set_xlabel("k")
        ax_conv.set_ylabel("F(x)")
        ax_conv.grid(True, alpha=0.3)

        figure.tight_layout()
        self._canvas(widgets).draw()

    @staticmethod
    def _entry(widgets: dict[str, object], key: str) -> ttk.Entry:
        return widgets[key]  # type: ignore[return-value]

    @staticmethod
    def _output(widgets: dict[str, object]) -> scrolledtext.ScrolledText:
        return widgets["output"]  # type: ignore[return-value]

    @staticmethod
    def _figure(widgets: dict[str, object]) -> plt.Figure:
        return widgets["figure"]  # type: ignore[return-value]

    @staticmethod
    def _canvas(widgets: dict[str, object]) -> FigureCanvasTkAgg:
        return widgets["canvas"]  # type: ignore[return-value]


def main() -> None:
    root = tk.Tk()
    Lr3Window(root)
    root.mainloop()
