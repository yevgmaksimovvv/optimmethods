"""Главная точка входа для ЛР5.

По умолчанию запускает базовую постановку, сохраняет отчётные артефакты и
печатает сводную таблицу в stdout.
"""

from __future__ import annotations

import argparse
import logging
import uuid

from lr5.application.artifacts import BarrierArtifactsStore
from lr5.application.services import (
    DEFAULT_BARRIER_KIND,
    build_method_config,
    parse_vector,
    run_barrier_method,
    variant_2_problem,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ЛР5: метод барьерных функций")
    parser.add_argument("--start-point", default="0;0", help="Стартовая точка в формате x1;x2")
    parser.add_argument(
        "--mu0",
        type=float,
        default=10.0,
        help="Начальное значение mu",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Коэффициент уменьшения mu, 0 < beta < 1",
    )
    parser.add_argument(
        "--epsilon-outer",
        type=float,
        default=1e-3,
        help="Точность внешнего цикла",
    )
    parser.add_argument(
        "--max-outer-iterations",
        type=int,
        default=20,
        help="Защитный лимит внешних итераций",
    )
    parser.add_argument(
        "--barrier-kind",
        choices=("reciprocal", "log"),
        default=DEFAULT_BARRIER_KIND,
        help="Тип барьерной функции",
    )
    parser.add_argument("--inner-epsilon", type=float, default=1e-4, help="Точность внутреннего решателя")
    parser.add_argument("--inner-max-iterations", type=int, default=200, help="Лимит итераций внутреннего решателя")
    return parser


def _print_result(trace_id: str, report_dir: str, result) -> None:
    print(f"trace_id: {trace_id}")
    print(f"report_dir: {report_dir}")
    print(f"start: {result.start_point}")
    print(f"status: {result.status}")
    print(
        "outer_config: "
        f"mu0={result.config.mu0:.10g}, beta={result.config.beta:.10g}, "
        f"epsilon_outer={result.config.epsilon_outer:.10g}, "
        f"max_outer_iterations={result.config.max_outer_iterations}"
    )
    if result.last_valid_outer_iteration is not None:
        iteration = result.last_valid_outer_iteration
        print(f"last_valid: {iteration.x_mu_k}")
        print(f"F(last_valid): {iteration.objective_value:.10g}")
        print(f"g(last_valid): {iteration.constraints_values}")
        print(f"mu(last_valid): {iteration.mu_k:.10g}")
    else:
        print("last_valid: —")
    print(f"success: {result.success}")
    print(f"stop_reason: {result.stop_reason}")
    print()
    constraint_headers = " | ".join(f"g{i + 1}(x)" for i in range(len(result.problem.constraints)))
    print(f"k | mu_k | x_mu_k | F(x) | M(x) | mu*M | Theta | {constraint_headers} | inner_N")
    for iteration in result.iterations:
        constraint_values = " | ".join(f"{value:.10g}" for value in iteration.constraints_values)
        print(
            f"{iteration.k} | {iteration.mu_k:.6g} | {iteration.x_mu_k} | "
            f"{iteration.objective_value:.10g} | {iteration.barrier_metric:.10g} | "
            f"{iteration.barrier_metric_term:.10g} | {iteration.theta_value:.10g} | "
            f"{constraint_values} | "
            f"{iteration.inner_result.iterations_count}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    problem = variant_2_problem()
    config = build_method_config(
        mu0=args.mu0,
        beta=args.beta,
        epsilon_outer=args.epsilon_outer,
        max_outer_iterations=args.max_outer_iterations,
        barrier_kind=args.barrier_kind,
        inner_epsilon=args.inner_epsilon,
        inner_max_iterations=args.inner_max_iterations,
    )
    start_point = parse_vector(args.start_point)

    result = run_barrier_method(problem, start_point, config)
    trace_id = uuid.uuid4().hex[:12]
    report_dir = BarrierArtifactsStore().save_result(result, trace_id=trace_id)
    _print_result(trace_id, str(report_dir), result)


if __name__ == "__main__":
    main()
