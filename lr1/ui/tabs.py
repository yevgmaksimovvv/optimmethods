"""Вкладки правой части интерфейса и их внутренние helpers.

Здесь сосредоточена только логика отображения:
- таблицы сводки;
- дерево итераций;
- список прогонов серии;
- область с графиком.

Модуль намеренно не запускает расчёты сам, а лишь принимает готовый `RunReport`
и раскладывает его по виджетам.
"""

from typing import Callable, Iterable, Optional, Sequence, Tuple

from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from lr1.application.viewmodels import (
    SummaryViewModel,
    build_grid_run_tooltip,
    build_iteration_rows,
    build_plot_context,
    build_summary_view_model,
    format_float,
)
from lr1.domain.models import GridRunResult, RunReport, SearchResult
from lr1.domain.search import METHOD_SPECS
from lr1.ui.workers import PlotCanvas


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
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.NoSelection)
    table.setFocusPolicy(Qt.NoFocus)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setHighlightSections(False)
    table.setWordWrap(True)
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

        self.summary_stack = QStackedWidget()
        layout.addWidget(self.summary_stack)

        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setContentsMargins(16, 24, 16, 0)
        empty_layout.setSpacing(0)

        empty_card = QWidget()
        empty_card.setObjectName("SummaryEmptyCard")
        empty_card.setMaximumWidth(700)
        empty_card.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        empty_card_layout = QVBoxLayout(empty_card)
        empty_card_layout.setContentsMargins(30, 24, 30, 24)
        empty_card_layout.setSpacing(18)

        empty_title = QLabel("Пока нет результатов")
        empty_title.setObjectName("SummaryEmptyTitle")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_card_layout.addWidget(empty_title)

        empty_text = QLabel(
            "Слева выбери функцию, диапазон и метод.\n"
            "После запуска здесь появятся результаты и сравнение методов."
        )
        empty_text.setObjectName("SummaryEmptyText")
        empty_text.setAlignment(Qt.AlignCenter)
        empty_text.setWordWrap(True)
        empty_card_layout.addWidget(empty_text)

        self.summary_empty_label = QLabel("Рассчитать для текущих параметров или запусти серию расчётов.")
        self.summary_empty_label.setObjectName("SectionHint")
        self.summary_empty_label.setAlignment(Qt.AlignCenter)
        self.summary_empty_label.setWordWrap(True)
        empty_card_layout.addWidget(self.summary_empty_label)

        empty_row = QHBoxLayout()
        empty_row.addStretch(1)
        empty_row.addWidget(empty_card, 1)
        empty_row.addStretch(1)
        empty_layout.addLayout(empty_row)
        empty_layout.addStretch(1)
        self.summary_stack.addWidget(empty_page)

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
        self.summary_stack.addWidget(summary_scroll)

    def populate(self, report: Optional[RunReport]) -> None:
        """Заполняет вкладку данными отчёта или возвращает её в пустое состояние."""
        if report is None:
            self.summary_stack.setCurrentIndex(0)
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
        self.summary_stack.setCurrentIndex(1)
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
        self.table_method_buttons = []

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
        card.setProperty("selected", False)

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

    def _refresh_widget_style(self, widget: QWidget) -> None:
        """Принудительно обновляет стиль после смены динамических свойств Qt."""
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _sync_grid_run_card_selection(self) -> None:
        """Синхронизирует подсветку карточек с текущим выбранным элементом списка."""
        current_row = self.grid_run_list.currentRow()
        for row_index in range(self.grid_run_list.count()):
            item = self.grid_run_list.item(row_index)
            widget = self.grid_run_list.itemWidget(item)
            if widget is None:
                continue
            widget.setProperty("selected", row_index == current_row)
            self._refresh_widget_style(widget)

    def _handle_grid_run_change(self, row: int) -> None:
        """Реагирует на смену выбранного прогона серии."""
        self._sync_grid_run_card_selection()
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

        for method_key in report.method_keys:
            button = QRadioButton(METHOD_SPECS[method_key].title)
            button.setProperty("choice_value", method_key)
            button.toggled.connect(self.on_method_change)
            self.table_method_buttons_layout.addWidget(button)
            self.table_method_buttons.append(button)
            if method_key == selected_method_key:
                button.setChecked(True)
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
        if runs:
            self.grid_run_list.setCurrentRow(min(selected_index, len(runs) - 1))
        self._sync_grid_run_card_selection()
        _fit_list_height(self.grid_run_list)
        self.grid_run_list.blockSignals(False)

    def populate_iterations(self, result: Optional[SearchResult]) -> None:
        """Показывает итерации выбранного результата в дереве.

        Все значения показываются в нейтральном цвете, а выбранная на шаге
        пара `λ/f(λ)` или `μ/f(μ)` выделяется акцентом.
        """
        self.iterations_tree.clear()
        if result is None:
            _fit_tree_height(self.iterations_tree)
            return

        header = self.iterations_tree.header()
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

    def clear(self) -> None:
        """Очищает все элементы вкладки до исходного пустого состояния."""
        self.iterations_tree.clear()
        self.grid_run_list.clear()
        self.grid_run_list.hide()
        self.grid_runs_caption.hide()
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
