"""Вкладки правой части интерфейса и их внутренние helpers.

Здесь сосредоточена только логика отображения:
- таблицы сводки;
- дерево итераций;
- список прогонов серии;
- область с графиком.

Модуль намеренно не запускает расчёты сам, а лишь принимает готовый `RunReport`
и раскладывает его по виджетам.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from matplotlib.figure import Figure
from optim_core.ui.controls_builder import create_choice_chip_grid
from optim_core.ui.table_widgets import configure_data_table
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lr1.application.analysis import resolve_method_keys
from lr1.domain.functions import analytic_comment
from lr1.domain.models import GridRunResult, ReferencePoint, RunReport, SearchResult, SkippedRun
from lr1.domain.search import METHOD_SPECS
from lr1.ui.workers import PlotCanvas


@dataclass(frozen=True)
class TableData:
    """Готовые данные одной таблицы: заголовки и строки."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SummaryViewModel:
    """Полный набор данных для вкладки `Сводка`."""

    results_table: TableData
    reference_table: TableData
    skipped_table: TableData
    analytic_note: str
    observations: tuple[str, ...]
    show_reference: bool
    show_skipped: bool
    show_notes: bool


@dataclass(frozen=True)
class IterationRowViewModel:
    """Подготовленная строка таблицы итераций с признаком победившей стороны."""

    texts: tuple[str, ...]
    left_wins: bool


def format_float(value: float, digits: int = 6) -> str:
    """Единообразно форматирует вещественное число для интерфейса."""
    return f"{value:.{digits}f}"


def format_interval(interval: tuple[float, float]) -> str:
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


def result_error_pair(report: RunReport, x_value: float, f_value: float) -> tuple[str, str]:
    """Считает ошибки относительно теоретического ориентира, если он известен."""
    if report.reference_point is None:
        return "—", "—"
    return (
        format_float(abs(x_value - report.reference_point.x)),
        format_float(abs(f_value - report.reference_point.f)),
    )


def _describe_reference_source(source: str) -> str:
    """Преобразует короткий источник ориентира в читабельный русский текст."""
    if source == "левая граница":
        return "на левой границе"
    if source == "правая граница":
        return "на правой границе"
    return source


def _format_count_ru(value: int, one: str, few: str, many: str) -> str:
    """Выбирает правильную форму русского слова по числу."""
    mod10 = value % 10
    mod100 = value % 100
    if mod10 == 1 and mod100 != 11:
        return one
    if 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
        return few
    return many


def build_grid_observations(
    kind: str,
    method_key: str,
    successful_runs: Sequence[GridRunResult],
    skipped_runs: Sequence[SkippedRun],
    reference_point: ReferencePoint | None,
) -> tuple[str, ...]:
    """Формирует наблюдения и выводы для режима серии расчётов."""
    del kind
    lines: list[str] = []

    if reference_point is not None:
        source = getattr(reference_point, "source")
        x_value = getattr(reference_point, "x")
        f_value = getattr(reference_point, "f")
        if source in {"левая граница", "правая граница"}:
            lines.append(
                "Экстремум на этом интервале граничный: "
                f"теоретически он достигается {_describe_reference_source(source)} "
                f"(x = {x_value:.6f}, f(x) = {f_value:.6f})."
            )
        else:
            lines.append(
                f"Экстремум на этом интервале внутренний: теоретически x = {x_value:.6f}, "
                f"f(x) = {f_value:.6f}."
            )

    if skipped_runs:
        skipped_count = len(skipped_runs)
        combination_word = _format_count_ru(skipped_count, "комбинация", "комбинации", "комбинаций")
        lines.append(
            f"Пропущено {skipped_count} невалидные {combination_word} параметров. "
            "Это не ошибка метода, а ограничение на допустимые значения ε и l."
        )

    if successful_runs:
        best_run = min(successful_runs, key=lambda item: item.result.func_evals)
        lines.append(
            f"Самый экономный запуск по числу вызовов функции: {METHOD_SPECS[best_run.method_key].title} "
            f"при ε={best_run.eps}, l={best_run.l} (вызовов: {best_run.result.func_evals})."
        )

        for current_method_key in resolve_method_keys(method_key):
            method_runs = [item for item in successful_runs if item.method_key == current_method_key]
            if len(method_runs) < 2:
                continue

            avg_by_l: dict[float, float] = {}
            l_values = tuple(sorted({item.l for item in method_runs}))
            for l_value in l_values:
                evals = [item.result.func_evals for item in method_runs if item.l == l_value]
                if evals:
                    avg_by_l[l_value] = sum(evals) / len(evals)
            if len(avg_by_l) == 2 and avg_by_l[min(avg_by_l)] > avg_by_l[max(avg_by_l)]:
                lines.append(
                    f"Для {METHOD_SPECS[current_method_key].title} уменьшение l "
                    f"с {max(avg_by_l):g} до {min(avg_by_l):g} увеличивает среднее число вызовов функции."
                )

            avg_by_eps: dict[float, float] = {}
            eps_values = tuple(sorted({item.eps for item in method_runs}))
            for eps_value in eps_values:
                evals = [item.result.func_evals for item in method_runs if item.eps == eps_value]
                if evals:
                    avg_by_eps[eps_value] = sum(evals) / len(evals)
            if len(avg_by_eps) >= 2:
                eps_min = min(avg_by_eps)
                eps_max = max(avg_by_eps)
                if avg_by_eps[eps_min] > avg_by_eps[eps_max]:
                    lines.append(
                        f"Для {METHOD_SPECS[current_method_key].title} уменьшение ε "
                        f"с {eps_max:g} до {eps_min:g} повышает среднее число вызовов функции."
                    )
                elif abs(avg_by_eps[eps_min] - avg_by_eps[eps_max]) < 1e-9:
                    lines.append(
                        f"Для {METHOD_SPECS[current_method_key].title} в этой сетке число вызовов функции почти не зависит от ε."
                    )

    return tuple(lines)


def build_summary_view_model(report: RunReport) -> SummaryViewModel:
    """Строит все таблицы и текстовые блоки для вкладки `Сводка`."""
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


def build_iteration_rows(result: SearchResult) -> tuple[IterationRowViewModel, ...]:
    """Готовит строки для визуализации итерационного процесса метода."""
    rows: list[IterationRowViewModel] = []
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


def build_plot_context(report: RunReport, selected_method_key: str, selected_run: Optional[GridRunResult]) -> str:
    """Генерирует поясняющий текст над графиком."""
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
def _fit_table_height(table: QTableWidget) -> None:
    """Подгоняет высоту таблицы под ограниченное число видимых строк.

    Это позволяет показать компактную таблицу без лишнего пустого пространства
    и включать вертикальный скролл только тогда, когда строк действительно много.
    """
    max_visible_rows = int(table.property("max_visible_rows") or 6)
    visible_rows = min(table.rowCount(), max_visible_rows)
    header_height = table.horizontalHeader().height() if table.columnCount() else 0
    rows_height = sum(table.rowHeight(index) for index in range(visible_rows))
    frame_height = table.frameWidth() * 2
    base_padding = 10

    if table.rowCount() == 0:
        empty_height = max(header_height + frame_height + 72, 120)
        table.setFixedHeight(empty_height)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return

    height = header_height + rows_height + frame_height + base_padding
    table.setFixedHeight(min(max(height, 96), 320))
    table.setVerticalScrollBarPolicy(
        Qt.ScrollBarAsNeeded if table.rowCount() > max_visible_rows else Qt.ScrollBarAlwaysOff
    )


def _fit_list_height(widget: QListWidget) -> None:
    """Подбирает высоту списка карточек под текущее число элементов."""
    max_visible_rows = int(widget.property("max_visible_rows") or 5)
    if widget.count() == 0:
        widget.setFixedHeight(80)
        return
    row_height = widget.sizeHintForRow(0)
    if row_height <= 0:
        row_height = 28
    visible_rows = min(widget.count(), max_visible_rows)
    frame_height = widget.frameWidth() * 2
    height = (row_height * visible_rows) + frame_height + 12
    widget.setFixedHeight(min(max(height, 72), 240))


def _fit_tree_height(tree: QTreeWidget) -> None:
    """Ограничивает высоту дерева итераций, сохраняя читаемость интерфейса."""
    max_visible_rows = int(tree.property("max_visible_rows") or 14)
    visible_rows = min(tree.topLevelItemCount(), max_visible_rows)
    header_height = tree.header().height() if tree.columnCount() else 0
    rows_height = 0
    for index in range(visible_rows):
        rows_height += tree.sizeHintForIndex(tree.indexFromItem(tree.topLevelItem(index), 0)).height()
    frame_height = tree.frameWidth() * 2

    if tree.topLevelItemCount() == 0:
        tree.setFixedHeight(max(header_height + frame_height + 96, 140))
        tree.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return

    height = header_height + rows_height + frame_height + 12
    tree.setFixedHeight(min(max(height, 140), 520))
    tree.setVerticalScrollBarPolicy(
        Qt.ScrollBarAsNeeded if tree.topLevelItemCount() > max_visible_rows else Qt.ScrollBarAlwaysOff
    )


def _resize_tree_columns_proportionally(tree: QTreeWidget) -> None:
    """Распределяет ширину дерева по всем колонкам пропорционально содержимому."""
    if tree.columnCount() == 0:
        return

    header = tree.header()
    header.setStretchLastSection(False)
    for index in range(tree.columnCount()):
        header.setSectionResizeMode(index, QHeaderView.Interactive)
        tree.resizeColumnToContents(index)

    base_widths = [max(tree.columnWidth(index), 44) for index in range(tree.columnCount())]
    available_width = tree.viewport().width() or sum(base_widths)
    total_width = sum(base_widths)
    if total_width <= 0:
        return

    scaled_widths = [max(44, int(round(width * available_width / total_width))) for width in base_widths]
    current_total = sum(scaled_widths)
    difference = available_width - current_total

    if difference > 0:
        order = sorted(range(tree.columnCount()), key=lambda index: base_widths[index], reverse=True)
        for step in range(difference):
            scaled_widths[order[step % len(order)]] += 1
    elif difference < 0:
        shrinkable = [index for index, width in enumerate(scaled_widths) if width > 44]
        step = 0
        while difference < 0 and shrinkable:
            index = shrinkable[step % len(shrinkable)]
            if scaled_widths[index] > 44:
                scaled_widths[index] -= 1
                difference += 1
            else:
                shrinkable.remove(index)
                step -= 1
            step += 1

    for index, width in enumerate(scaled_widths):
        header.resizeSection(index, width)


def _create_table_widget() -> QTableWidget:
    """Создаёт базовую таблицу с общей конфигурацией стиля для вкладки `Сводка`."""
    table = QTableWidget()
    table.setProperty("variant", "report")
    configure_data_table(
        table,
        min_row_height=31,
        allow_selection=False,
        allow_editing=False,
        word_wrap=True,
    )
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setHighlightSections(False)
    return table


def _column_alignment(header_text: str, value: str) -> Qt.AlignmentFlag:
    """Выбирает выравнивание колонки по смыслу её содержимого."""
    del value
    if header_text in {"Метод", "Параметр", "Значение", "Источник", "Причина", "Финальный интервал"}:
        return Qt.AlignLeft | Qt.AlignVCenter
    return Qt.AlignRight | Qt.AlignVCenter


def _set_table_data(table: QTableWidget, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Полностью перезаписывает содержимое таблицы и подгоняет её геометрию."""
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            item = QTableWidgetItem(value)
            header_text = headers[col_index] if col_index < len(headers) else ""
            item.setTextAlignment(_column_alignment(header_text, value))
            table.setItem(row_index, col_index, item)
    header = table.horizontalHeader()
    for index in range(len(headers)):
        header.setSectionResizeMode(index, QHeaderView.ResizeToContents)
    if headers:
        header.setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)
    table.resizeRowsToContents()
    _fit_table_height(table)


class SummaryTab(QWidget):
    """Вкладка со сводными таблицами, аналитическим ориентиром и выводами."""
    def __init__(self):
        """Создаёт пустое состояние вкладки и все её секции."""
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setFrameShape(QScrollArea.NoFrame)
        summary_content = QWidget()
        summary_layout = QVBoxLayout(summary_content)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(12)

        self.summary_results_box = QGroupBox("Результаты")
        results_layout = QVBoxLayout(self.summary_results_box)
        self.summary_results_table = _create_table_widget()
        self.summary_results_table.setProperty("max_visible_rows", 6)
        results_layout.addWidget(self.summary_results_table)
        summary_layout.addWidget(self.summary_results_box)

        self.summary_reference_box = QGroupBox("Теоретический ориентир")
        reference_layout = QVBoxLayout(self.summary_reference_box)
        self.summary_reference_table = _create_table_widget()
        self.summary_reference_table.setProperty("max_visible_rows", 4)
        self.summary_analytic_label = QLabel("—")
        self.summary_analytic_label.setObjectName("SectionHint")
        self.summary_analytic_label.setWordWrap(True)
        reference_layout.addWidget(self.summary_reference_table)
        reference_layout.addWidget(self.summary_analytic_label)
        summary_layout.addWidget(self.summary_reference_box)

        self.summary_skipped_box = QGroupBox("Пропущенные наборы параметров")
        skipped_layout = QVBoxLayout(self.summary_skipped_box)
        self.summary_skipped_table = _create_table_widget()
        self.summary_skipped_table.setProperty("max_visible_rows", 5)
        skipped_layout.addWidget(self.summary_skipped_table)
        summary_layout.addWidget(self.summary_skipped_box)

        self.summary_notes_box = QGroupBox("Выводы")
        notes_layout = QVBoxLayout(self.summary_notes_box)
        self.summary_notes_list = QListWidget()
        self.summary_notes_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.summary_notes_list.setFocusPolicy(Qt.NoFocus)
        self.summary_notes_list.setProperty("max_visible_rows", 5)
        notes_layout.addWidget(self.summary_notes_list)
        summary_layout.addWidget(self.summary_notes_box)
        summary_layout.addStretch(1)

        summary_scroll.setWidget(summary_content)
        layout.addWidget(summary_scroll)

    def populate(self, report: Optional[RunReport]) -> None:
        """Заполняет вкладку данными отчёта или возвращает её в пустое состояние."""
        if report is None:
            self.summary_results_box.hide()
            _set_table_data(self.summary_results_table, ("Результат",), ())
            _set_table_data(self.summary_reference_table, ("Параметр", "Значение"), ())
            _set_table_data(self.summary_skipped_table, ("Метод", "ε", "l", "Причина"), ())
            self.summary_reference_box.hide()
            self.summary_skipped_box.hide()
            self.summary_notes_box.hide()
            self.summary_notes_list.clear()
            self.summary_analytic_label.setText("—")
            return

        view_model = build_summary_view_model(report)
        self.summary_results_box.show()
        _set_table_data(self.summary_results_table, view_model.results_table.headers, view_model.results_table.rows)
        _set_table_data(self.summary_reference_table, view_model.reference_table.headers, view_model.reference_table.rows)
        _set_table_data(self.summary_skipped_table, view_model.skipped_table.headers, view_model.skipped_table.rows)
        self.summary_analytic_label.setText(view_model.analytic_note)

        self.summary_reference_box.setVisible(view_model.show_reference)
        self.summary_skipped_box.setVisible(view_model.show_skipped)
        self.summary_notes_box.setVisible(view_model.show_notes)

        self.summary_notes_list.clear()
        for note in view_model.observations:
            self.summary_notes_list.addItem(note)
        _fit_list_height(self.summary_notes_list)


class IterationsTab(QWidget):
    """Вкладка с подробной историей итераций выбранного метода."""
    def __init__(self, on_grid_run_change: Callable[[int], None], on_method_change: Callable[[], None]):
        """Строит переключатель методов, список прогонов серии и дерево итераций."""
        super().__init__()
        self.on_grid_run_change = on_grid_run_change
        self.on_method_change = on_method_change
        self.table_method_buttons: list[QPushButton] = []
        self.table_method_group = QButtonGroup(self)
        self.table_method_group.setExclusive(True)

        layout = QVBoxLayout(self)
        self.table_method_box = QGroupBox("Итерации выбранного метода")
        method_layout = QHBoxLayout(self.table_method_box)
        method_layout.setContentsMargins(10, 10, 10, 10)
        method_layout.setSpacing(10)
        self.table_method_buttons_layout = method_layout
        layout.addWidget(self.table_method_box)

        self.iterations_tree = QTreeWidget()
        self.iterations_tree.setObjectName("IterationsTree")
        self.iterations_tree.setColumnCount(7)
        self.iterations_tree.setHeaderLabels(("k", "a", "b", "λ", "μ", "f(λ)", "f(μ)"))
        self.iterations_tree.setAlternatingRowColors(True)
        self.iterations_tree.setRootIsDecorated(False)
        self.iterations_tree.setUniformRowHeights(True)
        self.iterations_tree.setIndentation(0)
        self.iterations_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.iterations_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.iterations_tree.setProperty("max_visible_rows", 14)
        self.iterations_tree_holder = QWidget()
        iterations_tree_holder_layout = QHBoxLayout(self.iterations_tree_holder)
        iterations_tree_holder_layout.setContentsMargins(0, 0, 0, 0)
        iterations_tree_holder_layout.setSpacing(0)
        iterations_tree_holder_layout.addWidget(self.iterations_tree, 1)

        self.grid_runs_caption = QLabel("Прогоны серии")
        self.grid_runs_caption.setObjectName("SectionCaption")
        self.grid_runs_caption.hide()
        self.grid_run_list = QListWidget()
        self.grid_run_list.setObjectName("GridRunList")
        self.grid_run_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid_run_list.setFocusPolicy(Qt.StrongFocus)
        self.grid_run_list.setProperty("max_visible_rows", 4)
        self.grid_run_list.hide()
        self.grid_run_list.currentRowChanged.connect(self._handle_grid_run_change)

        layout.addWidget(self.grid_runs_caption)
        layout.addWidget(self.grid_run_list)
        layout.addWidget(self.iterations_tree_holder)
        self.final_result_card = QWidget()
        self.final_result_card.setObjectName("FinalResultCard")
        final_layout = QVBoxLayout(self.final_result_card)
        final_layout.setContentsMargins(14, 12, 14, 12)
        final_layout.setSpacing(8)

        self.final_result_title = QLabel("Итог расчёта")
        self.final_result_title.setObjectName("FinalResultTitle")
        final_layout.addWidget(self.final_result_title)

        final_grid = QGridLayout()
        final_grid.setContentsMargins(0, 0, 0, 0)
        final_grid.setHorizontalSpacing(12)
        final_grid.setVerticalSpacing(6)

        self.final_interval_caption = QLabel("Финальный интервал:")
        self.final_interval_caption.setProperty("role", "final-result-caption")
        self.final_interval_value = QLabel("—")
        self.final_interval_value.setProperty("role", "final-result-value")

        self.final_x_caption = QLabel("Оптимальный x* =")
        self.final_x_caption.setProperty("role", "final-result-caption")
        self.final_x_value = QLabel("—")
        self.final_x_value.setProperty("role", "final-result-value")

        self.final_f_caption = QLabel("Оптимальное значение F(x*) =")
        self.final_f_caption.setProperty("role", "final-result-caption")
        self.final_f_value = QLabel("—")
        self.final_f_value.setProperty("role", "final-result-value")

        final_grid.addWidget(self.final_interval_caption, 0, 0)
        final_grid.addWidget(self.final_interval_value, 0, 1)
        final_grid.addWidget(self.final_x_caption, 1, 0)
        final_grid.addWidget(self.final_x_value, 1, 1)
        final_grid.addWidget(self.final_f_caption, 2, 0)
        final_grid.addWidget(self.final_f_value, 2, 1)
        final_grid.setColumnStretch(1, 1)
        final_layout.addLayout(final_grid)

        self._set_final_result_text(None)
        layout.addWidget(self.final_result_card)
        layout.addStretch(1)

    def resizeEvent(self, event) -> None:
        """При изменении размера перераспределяет ширину всех колонок таблицы."""
        super().resizeEvent(event)
        if self.iterations_tree.topLevelItemCount():
            _resize_tree_columns_proportionally(self.iterations_tree)

    def _create_grid_run_metric(self, caption: str, value: str) -> QWidget:
        """Создаёт одну метрику внутри карточки прогона серии."""
        cell = QWidget()
        cell.setObjectName("GridRunMetric")
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        caption_label = QLabel(caption)
        caption_label.setProperty("role", "grid-run-metric-caption")
        caption_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        value_label = QLabel(value)
        value_label.setProperty("role", "grid-run-metric-value")
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        value_label.setMinimumHeight(30)

        layout.addWidget(caption_label)
        layout.addWidget(value_label)
        return cell

    def _create_grid_run_card(self, run: GridRunResult) -> QWidget:
        """Собирает визуальную карточку одного запуска в режиме серии."""
        card = QWidget()
        card.setObjectName("GridRunCard")

        layout = QHBoxLayout(card)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(28)

        metrics = (
            ("ε", f"{run.eps:g}", 0, 92),
            ("l", f"{run.l:g}", 0, 78),
            ("Вызовы", str(run.result.func_evals), 0, 116),
            ("x*", format_float(run.result.x_opt), 1, 172),
        )
        for caption, value, stretch, min_width in metrics:
            metric = self._create_grid_run_metric(caption, value)
            metric.setMinimumWidth(min_width)
            layout.addWidget(metric, stretch)

        return card

    def _handle_grid_run_change(self, row: int) -> None:
        """Реагирует на смену выбранного прогона серии."""
        self.on_grid_run_change(row)

    def rebuild_method_buttons(self, report: Optional[RunReport], selected_method_key: str) -> None:
        """Перестраивает радиокнопки методов под текущий отчёт.

        Набор кнопок зависит от того, какие методы реально выполнились
        и попали в `report.method_keys`.
        """
        while self.table_method_buttons_layout.count():
            item = self.table_method_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.table_method_buttons = []

        if report is None:
            return

        options = [(METHOD_SPECS[method_key].title, method_key) for method_key in report.method_keys]
        holder, buttons = create_choice_chip_grid(
            group=self.table_method_group,
            options=options,
            columns=max(len(options), 1),
            horizontal_spacing=10,
            vertical_spacing=10,
        )
        self.table_method_buttons = buttons
        for button in self.table_method_buttons:
            button.toggled.connect(self.on_method_change)
        self.table_method_buttons_layout.addWidget(holder)
        selected_found = False
        for button in self.table_method_buttons:
            if str(button.property("choice_value")) == selected_method_key:
                button.setChecked(True)
                selected_found = True
                break
        if not selected_found and self.table_method_buttons:
            self.table_method_buttons[0].setChecked(True)
        self.table_method_buttons_layout.addStretch(1)

    def populate_grid_runs(self, report: Optional[RunReport], selected_method_key: str, selected_index: int) -> None:
        """Заполняет список прогонов для режима серии расчётов."""
        self.grid_run_list.blockSignals(True)
        self.grid_run_list.clear()
        if report is None or report.mode != "grid":
            self.grid_runs_caption.hide()
            self.grid_run_list.hide()
            _fit_list_height(self.grid_run_list)
            self.grid_run_list.blockSignals(False)
            return

        runs = report.grid_runs_by_method.get(selected_method_key, ())
        for run in runs:
            item = QListWidgetItem()
            item.setToolTip(build_grid_run_tooltip(run))
            card = self._create_grid_run_card(run)
            item.setSizeHint(card.sizeHint())
            self.grid_run_list.addItem(item)
            self.grid_run_list.setItemWidget(item, card)

        self.grid_runs_caption.setVisible(bool(runs))
        self.grid_run_list.setVisible(bool(runs))
        _fit_list_height(self.grid_run_list)
        self.grid_run_list.blockSignals(False)
        if runs:
            self.grid_run_list.setCurrentRow(min(selected_index, len(runs) - 1))

    def populate_iterations(self, result: Optional[SearchResult]) -> None:
        """Показывает итерации выбранного результата в дереве.

        Все значения показываются в нейтральном цвете, а выбранная на шаге
        пара `λ/f(λ)` или `μ/f(μ)` выделяется акцентом.
        """
        self.iterations_tree.clear()
        if result is None:
            self._set_final_result_text(None)
            _fit_tree_height(self.iterations_tree)
            return

        header_item = self.iterations_tree.headerItem()
        header_item.setTextAlignment(0, Qt.AlignCenter)
        for index in range(1, self.iterations_tree.columnCount()):
            header_item.setTextAlignment(index, Qt.AlignRight | Qt.AlignVCenter)

        base_foreground = QBrush(QColor("#d7deea"))
        accent_foreground = QBrush(QColor("#f2e7c9"))

        for view_row in build_iteration_rows(result):
            item = QTreeWidgetItem(self.iterations_tree, list(view_row.texts))
            item.setTextAlignment(0, Qt.AlignCenter)
            for column in range(self.iterations_tree.columnCount()):
                item.setForeground(column, base_foreground)
            for column in range(1, self.iterations_tree.columnCount()):
                item.setTextAlignment(column, Qt.AlignRight | Qt.AlignVCenter)

            highlighted_columns = (3, 5) if view_row.left_wins else (4, 6)
            for column in highlighted_columns:
                item.setForeground(column, accent_foreground)

        _resize_tree_columns_proportionally(self.iterations_tree)
        _fit_tree_height(self.iterations_tree)
        self._set_final_result_text(result)

    def _set_final_result_text(self, result: Optional[SearchResult]) -> None:
        """Обновляет блок с финальными итогами текущего расчёта."""
        if result is None:
            self.final_interval_value.setText("—")
            self.final_x_value.setText("—")
            self.final_f_value.setText("—")
            return
        self.final_interval_value.setText(format_interval(result.interval_final))
        self.final_x_value.setText(format_float(result.x_opt))
        self.final_f_value.setText(format_float(result.f_opt))

    def clear(self) -> None:
        """Очищает все элементы вкладки до исходного пустого состояния."""
        self.iterations_tree.clear()
        self.grid_run_list.clear()
        self.grid_run_list.hide()
        self.grid_runs_caption.hide()
        self._set_final_result_text(None)
        self.rebuild_method_buttons(None, "")


class PlotTab(QWidget):
    """Вкладка для отображения готовой Matplotlib-фигуры и поясняющего текста."""
    def __init__(self):
        """Создаёт контейнер для графика, прокрутку и подписи состояния."""
        super().__init__()
        self.plot_canvas: Optional[PlotCanvas] = None

        layout = QVBoxLayout(self)
        self.plot_context_label = QLabel("")
        self.plot_context_label.setObjectName("SectionHint")
        self.plot_context_label.setWordWrap(True)
        self.plot_context_label.hide()
        layout.addWidget(self.plot_context_label)

        self.plot_state_label = QLabel("График появится после расчёта.")
        self.plot_state_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_state_label)

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setFrameShape(QScrollArea.NoFrame)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.plot_scroll.verticalScrollBar().setSingleStep(32)
        layout.addWidget(self.plot_scroll)

        self.plot_host = QWidget()
        self.plot_host_layout = QVBoxLayout(self.plot_host)
        self.plot_host_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_host_layout.setSpacing(0)
        self.plot_host_layout.setAlignment(Qt.AlignTop)
        self.plot_scroll.setWidget(self.plot_host)

    def show_placeholder(self, text: str) -> None:
        """Показывает текст-заглушку и убирает текущий график, если он есть."""
        self.plot_state_label.setText(text)
        self.plot_state_label.show()
        if text:
            self.plot_context_label.hide()
        if self.plot_canvas is not None:
            self.plot_host_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()
            self.plot_canvas = None

    def show_figure(self, figure: Figure, context_text: str) -> None:
        """Вставляет построенную фигуру во вкладку и обновляет поясняющий текст."""
        self.show_placeholder("")
        self.plot_state_label.hide()
        if context_text:
            self.plot_context_label.setText(context_text)
            self.plot_context_label.show()
        else:
            self.plot_context_label.hide()
        self.plot_canvas = PlotCanvas(figure, self.plot_host)
        _width_px, height_px = figure.get_size_inches() * figure.dpi
        self.plot_canvas.setMinimumWidth(0)
        self.plot_canvas.setFixedHeight(int(height_px))
        self.plot_host_layout.addWidget(self.plot_canvas)
        self.plot_canvas.draw()

    def context_text(self, report: RunReport, selected_method_key: str, selected_run: Optional[GridRunResult]) -> str:
        """Возвращает текстовый контекст для текущего состояния графика."""
        return build_plot_context(report, selected_method_key, selected_run)
