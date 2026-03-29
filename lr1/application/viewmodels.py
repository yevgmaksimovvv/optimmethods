"""Преобразование `RunReport` в данные, удобные для отображения.

UI не должен сам решать, как форматировать числа, интервалы и текстовые
пояснения. Этот модуль берёт доменные результаты и подготавливает из них
простые view-model структуры для таблиц, списков и графических подсказок.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from lr1.application.analysis import build_grid_observations
from lr1.domain.functions import analytic_comment
from lr1.domain.models import GridRunResult, RunReport, SearchResult
from lr1.domain.search import METHOD_SPECS


@dataclass(frozen=True)
class TableData:
    """Готовые данные одной таблицы: заголовки и строки."""
    headers: Tuple[str, ...]
    rows: Tuple[Tuple[str, ...], ...]


@dataclass(frozen=True)
class SummaryViewModel:
    """Полный набор данных для вкладки `Сводка`."""
    results_table: TableData
    reference_table: TableData
    skipped_table: TableData
    analytic_note: str
    observations: Tuple[str, ...]
    show_reference: bool
    show_skipped: bool
    show_notes: bool


@dataclass(frozen=True)
class IterationRowViewModel:
    """Подготовленная строка таблицы итераций с признаком победившей стороны."""
    texts: Tuple[str, ...]
    left_wins: bool


def format_float(value: float, digits: int = 6) -> str:
    """Единообразно форматирует вещественное число для интерфейса."""
    return f"{value:.{digits}f}"


def format_interval(interval: Tuple[float, float]) -> str:
    """Форматирует интервал в компактную строку `[a, b]`."""
    return f"[{format_float(interval[0], 5)}, {format_float(interval[1], 5)}]"


def format_optional_float(value: Optional[float]) -> str:
    """Форматирует число или подставляет тире для отсутствующего значения."""
    return "—" if value is None else format_float(value)


def format_iteration_value(value: float) -> str:
    """Задаёт формат чисел в таблице итераций."""
    return format_float(value, 5)


def iteration_left_wins(kind: str, left_value: float, right_value: float) -> bool:
    """Определяет, какая пробная точка считается лучшей на текущем шаге."""
    if kind == "max":
        return left_value >= right_value
    return left_value <= right_value


def result_error_pair(report: RunReport, x_value: float, f_value: float) -> Tuple[str, str]:
    """Считает ошибки относительно теоретического ориентира, если он известен."""
    if report.reference_point is None:
        return "—", "—"
    return (
        format_float(abs(x_value - report.reference_point.x)),
        format_float(abs(f_value - report.reference_point.f)),
    )


def build_summary_view_model(report: RunReport) -> SummaryViewModel:
    """Строит все таблицы и текстовые блоки для вкладки `Сводка`.

    Структура вкладки зависит от режима:
    - в одиночном запуске показывается по одной строке на метод;
    - в серии расчётов показываются все успешные прогоны по сетке.
    """
    result_rows: list[tuple[str, ...]] = []
    result_headers: tuple[str, ...]
    if report.mode == "grid":
        result_headers = (
            "Метод",
            "ε",
            "l",
            "x*",
            "f(x*)",
            "Итерации",
            "Вызовы функции",
            "Финальный интервал",
            "|Δx|",
            "|Δf|",
        )
        for method_key in report.method_keys:
            for run in report.grid_runs_by_method.get(method_key, ()):
                dx, df = result_error_pair(report, run.result.x_opt, run.result.f_opt)
                result_rows.append(
                    (
                        run.result.method,
                        f"{run.eps:g}",
                        f"{run.l:g}",
                        format_float(run.result.x_opt),
                        format_float(run.result.f_opt),
                        str(len(run.result.iterations)),
                        str(run.result.func_evals),
                        format_interval(run.result.interval_final),
                        dx,
                        df,
                    )
                )
    else:
        result_headers = (
            "Метод",
            "x*",
            "f(x*)",
            "Итерации",
            "Вызовы функции",
            "Финальный интервал",
            "|Δx|",
            "|Δf|",
        )
        for method_key in report.method_keys:
            result = report.results_by_method.get(method_key)
            if result is None:
                continue
            dx, df = result_error_pair(report, result.x_opt, result.f_opt)
            result_rows.append(
                (
                    result.method,
                    format_float(result.x_opt),
                    format_float(result.f_opt),
                    str(len(result.iterations)),
                    str(result.func_evals),
                    format_interval(result.interval_final),
                    dx,
                    df,
                )
            )

    reference_table = TableData(
        headers=("Параметр", "Значение"),
        rows=(
            ("x*", format_optional_float(report.reference_point.x if report.reference_point else None)),
            ("f(x*)", format_optional_float(report.reference_point.f if report.reference_point else None)),
            ("Источник", report.reference_point.source if report.reference_point else "—"),
        ),
    )

    skipped_rows = tuple(
        (
            METHOD_SPECS[item.method_key].title,
            f"{item.eps:g}",
            f"{item.l:g}",
            item.reason,
        )
        for item in report.skipped_runs
    )
    observations = (
        build_grid_observations(
            kind=report.kind,
            method_key=report.requested_method_key,
            successful_runs=tuple(run for runs in report.grid_runs_by_method.values() for run in runs),
            skipped_runs=report.skipped_runs,
            reference_point=report.reference_point,
        )
        if report.mode == "grid"
        else ()
    )
    analytic_note = analytic_comment(report.function_spec, report.interval, report.kind) or "—"

    return SummaryViewModel(
        results_table=TableData(headers=result_headers, rows=tuple(result_rows)),
        reference_table=reference_table,
        skipped_table=TableData(headers=("Метод", "ε", "l", "Причина"), rows=skipped_rows),
        analytic_note=analytic_note,
        observations=observations,
        show_reference=report.reference_point is not None or bool(analytic_note),
        show_skipped=bool(skipped_rows),
        show_notes=bool(observations),
    )


def build_iteration_rows(result: SearchResult) -> Tuple[IterationRowViewModel, ...]:
    """Готовит строки для визуализации итерационного процесса метода."""
    rows: List[IterationRowViewModel] = []
    for row in result.iterations:
        rows.append(
            IterationRowViewModel(
                texts=(
                    str(row.k),
                    format_iteration_value(row.a),
                    format_iteration_value(row.b),
                    format_iteration_value(row.lam),
                    format_iteration_value(row.mu),
                    format_iteration_value(row.f_lam),
                    format_iteration_value(row.f_mu),
                ),
                left_wins=iteration_left_wins(result.kind, row.f_lam, row.f_mu),
            )
        )
    return tuple(rows)


def build_grid_run_tooltip(run: GridRunResult) -> str:
    """Собирает короткую всплывающую подсказку для карточки прогона."""
    return (
        f"ε={run.eps:g}, l={run.l:g}, вызовов={run.result.func_evals}, "
        f"x*={format_float(run.result.x_opt)}"
    )


def build_plot_context(
    report: RunReport,
    selected_method_key: str,
    selected_run: Optional[GridRunResult],
) -> str:
    """Генерирует поясняющий текст над графиком.

    Этот текст подсказывает пользователю, что именно сейчас изображено:
    сравнение методов или конкретный выбранный прогон из серии.
    """
    if report.mode == "grid":
        if selected_run is None or not selected_method_key:
            return ""
        method_title = METHOD_SPECS[selected_method_key].title if selected_method_key in METHOD_SPECS else "выбранного метода"
        return (
            f"Сверху показана вся серия для метода {method_title}. "
            f"Снизу открыт выбранный прогон: ε={selected_run.eps:g}, l={selected_run.l:g}, "
            f"вызовов={selected_run.result.func_evals}, x*={format_float(selected_run.result.x_opt)}."
        )

    if len(report.results_by_method) > 1:
        return "На графике показано сравнение методов: каждый подграфик соответствует одному методу."
    return "На графике показан выбранный метод и его точки итераций."
