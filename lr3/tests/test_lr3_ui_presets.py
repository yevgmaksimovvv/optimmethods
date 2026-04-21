"""UI-smoke тесты preset-контракта ЛР3."""

from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication, QLabel

from lr3.application.services import build_config, run_conjugate, run_gradient
from lr3.domain.expression import analyze_local_extremum, compile_objective
from lr3.domain.models import IterationRecord, OptimizationResult
from lr3.ui.window import GradientMethodsWindow, RunPayload


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _coefficients(window: GradientMethodsWindow) -> tuple[str, ...]:
    return tuple(window.coefficient_table.item(row, 1).text() for row in range(window.coefficient_table.rowCount()))


def test_lr3_presets_fill_coefficients_and_custom_is_editable() -> None:
    _app()
    window = GradientMethodsWindow()

    assert set(window.preset_buttons) == {"gradient", "conjugate", "custom"}
    assert window.preset_buttons["gradient"].text() == "Градиентная"
    assert window.preset_buttons["conjugate"].text() == "Сопр. градиенты"
    assert window.preset_buttons["custom"].text() == "Пользовательская"
    assert set(window.goal_buttons) == {"min", "max"}
    assert window.goal_buttons["min"].text() == "Минимум"
    assert window.goal_buttons["max"].text() == "Максимум"
    assert _coefficients(window) == ("0", "1", "-2", "1", "-1", "1")
    assert not window.coefficient_table.isEnabled()
    assert "F(x₁, x₂) =" in window.formula_preview.text()
    assert "x₁²" in window.formula_preview.text()
    assert window.goal_buttons["min"].isChecked()

    window.preset_buttons["conjugate"].click()
    assert _coefficients(window) == ("-2", "-1", "-2", "-0.1", "0", "-100")
    assert not window.coefficient_table.isEnabled()
    assert window.start_x1_input.text() == "1"
    assert window.start_x2_input.text() == "1"

    window.preset_buttons["custom"].click()
    assert window.coefficient_table.isEnabled()
    window._last_payload = object()
    window.results_empty_stack.set_empty(False)
    window.coefficient_table.item(1, 1).setText("3")
    assert "3x₁" in window.formula_preview.text()
    assert window._last_payload is None
    assert window.results_empty_stack.currentIndex() == 0


def test_lr3_method_buttons_update_numeric_defaults() -> None:
    _app()
    window = GradientMethodsWindow()

    window.method_buttons["conjugate"].click()

    assert window.start_x1_input.text() == "1"
    assert window.start_x2_input.text() == "1"
    assert window.initial_step_input.text() == "0.2"
    assert window._selected_method_iterations() == "300"
    assert not hasattr(window, "max_iterations_input")
    assert not hasattr(window, "max_step_expansions_input")
    assert not hasattr(window, "min_step_input")
    assert not hasattr(window, "gradient_step_input")


def test_lr3_analysis_card_renders_symbolic_hessian() -> None:
    _app()
    window = GradientMethodsWindow()
    analysis = analyze_local_extremum("x1**2 + x2**2 - x1*x2 + x1 - 2*x2", (0.0, 0.0), goal="min")

    card = window._build_analysis_card(analysis=analysis)
    texts = [label.text() for label in card.findChildren(QLabel)]

    assert card.title() == "Аналитическая подготовка"
    assert all("Функция" not in text for text in texts)
    assert all("Точка запуска" not in text for text in texts)
    assert all("Что показывает теория" not in text for text in texts)
    assert all("Согласование с задачей" not in text for text in texts)
    assert all("Что удаётся строго" not in text for text in texts)
    assert any("∇f(x<sub>1</sub>, x<sub>2</sub>)" in text for text in texts)
    assert any("∇f(M<sub>0</sub>)" in text for text in texts)
    assert any("M*" in text for text in texts)
    assert any(text == "H" for text in texts)
    assert any("Вывод:" in text for text in texts)
    assert any("Локальный минимум" in text for text in texts)
    assert any("Согласование" in text for text in texts)


def test_lr3_plot_render_handles_unbounded_maximum_without_crashing() -> None:
    _app()
    window = GradientMethodsWindow()
    config = build_config(
        epsilon_raw="1e-6",
        max_iterations_raw="250",
        initial_step_raw="0.1",
        timeout_raw="2.0",
        goal_raw="max",
    )
    result, metrics = run_gradient(
        expression="x1**2 + x2**2 - x1*x2 + x1 - 2*x2",
        start_point=(0.0, 0.0),
        config=config,
    )

    payload = RunPayload(
        expression="x1**2 + x2**2 - x1*x2 + x1 - 2*x2",
        config=config,
        result=result,
        metrics=metrics,
    )

    window._render_plot(payload)

    assert window.canvas.figure.axes


def test_lr3_gradient_step_comment_names_the_accepted_step() -> None:
    _app()
    window = GradientMethodsWindow()
    result = OptimizationResult(
        method_name="gradient_ascent",
        start_point=(0.0, 0.0),
        optimum_point=(0.0, 0.0),
        optimum_value=0.0,
        iterations_count=1,
        records=(
            IterationRecord(
                k=0,
                point=(0.0, 0.0),
                value=0.0,
                gradient=(1.0, -2.0),
                step_size=0.4,
                gradient_step_decision="accepted_as_is",
            ),
        ),
        success=True,
        stop_reason="gradient_norm_reached",
    )

    note_down = window._gradient_step_note(
        IterationRecord(k=0, point=(0.0, 0.0), value=0.0, gradient=(1.0, -2.0), step_size=0.05, gradient_step_decision="accepted_after_reduction"),
        result,
    )
    note_same = window._gradient_step_note(
        IterationRecord(k=0, point=(0.0, 0.0), value=0.0, gradient=(1.0, -2.0), step_size=0.1, gradient_step_decision="accepted_as_is"),
        result,
    )
    note_unchanged = window._gradient_step_note(
        IterationRecord(k=1, point=(0.0, 0.0), value=0.0, gradient=(1.0, -2.0), step_size=0.4, gradient_step_decision="accepted_as_is"),
        result,
        previous_step_size=0.4,
    )
    note_grown = window._gradient_step_note(
        IterationRecord(k=1, point=(0.0, 0.0), value=0.0, gradient=(1.0, -2.0), step_size=0.8, gradient_step_decision="accepted_as_is"),
        result,
        previous_step_size=0.4,
    )
    note_reduced = window._gradient_step_note(
        IterationRecord(k=1, point=(0.0, 0.0), value=0.0, gradient=(1.0, -2.0), step_size=0.4, gradient_step_decision="accepted_as_is"),
        result,
        previous_step_size=0.8,
    )
    note_stop = window._gradient_step_note(
        IterationRecord(k=0, point=(0.0, 0.0), value=0.0, gradient=(1.0, -2.0), step_size=0.0, gradient_step_decision="precision_reached"),
        result,
    )

    assert note_down == "Шаг h = 0.05 получен после уменьшения шага и принят."
    assert note_same == "Шаг h = 0.1 выбран сразу и принят."
    assert note_unchanged == "Шаг h = 0.4 не изменился."
    assert note_grown == "Шаг h = 0.8 увеличился."
    assert note_reduced == "Шаг h = 0.4 уменьшился."
    assert note_stop == "Достигнута требуемая точность, переход не выполнялся."


def test_lr3_gradient_iteration_card_marks_h_as_accepted_step() -> None:
    _app()
    window = GradientMethodsWindow()
    config = build_config(
        epsilon_raw="1e-5",
        max_iterations_raw="250",
        initial_step_raw="0.1",
        timeout_raw="2.0",
        goal_raw="min",
    )
    result, metrics = run_gradient(
        expression="x1**2 + x2**2 - x1*x2 + x1 - 2*x2",
        start_point=(0.0, 0.0),
        config=config,
    )
    payload = RunPayload(
        expression="x1**2 + x2**2 - x1*x2 + x1 - 2*x2",
        config=config,
        result=result,
        metrics=metrics,
    )

    cards = window._build_gradient_iteration_cards(payload, compile_objective(payload.expression))
    # Карточка должна явно называть h_k принятым шагом.
    texts = [label.text() for card in cards for label in card.findChildren(QLabel)]

    assert any("Принятый шаг h<sub>k</sub>" in text for text in texts)
    assert any("Шаг h =" in text for text in texts)
    assert any(("выбран сразу и принят" in text) or ("получен после уменьшения шага и принят" in text) for text in texts)


def test_lr3_conjugate_iteration_cards_explain_cycle_transition_and_direction_update() -> None:
    _app()
    window = GradientMethodsWindow()
    config = build_config(
        epsilon_raw="1e-6",
        max_iterations_raw="300",
        initial_step_raw="0.2",
        timeout_raw="2.0",
        goal_raw="max",
    )
    result, metrics = run_conjugate(
        expression="-2 - x1 - 2*x2 - 0.1*x1**2 - 100*x2**2",
        start_point=(1.0, 1.0),
        config=config,
    )
    payload = RunPayload(
        expression="-2 - x1 - 2*x2 - 0.1*x1**2 - 100*x2**2",
        config=config,
        result=result,
        metrics=metrics,
    )

    cards = window._build_conjugate_iteration_cards(payload)
    texts = [label.text() for card in cards for label in card.findChildren(QLabel)]

    assert any("В начале цикла y<sub>1</sub> = x<sub>k</sub>." in text for text in texts)
    assert any("Внутри цикла x<sub>k</sub> фиксирован, меняется только y<sub>j</sub>." in text for text in texts)
    assert any("s<sub>1</sub> = ∇F(y<sub>1</sub>)" in text for text in texts)
    assert any("s<sub>j</sub> = ∇F(y<sub>j</sub>) + β<sub>j</sub>·s<sub>j-1</sub>" in text for text in texts)
    assert any("x<sub>2</sub> = y<sub>3</sub>" in text for text in texts)
    assert any("Это последний шаг цикла: после него x<sub>k+1</sub> = y<sub>j+1</sub>." in text for text in texts)
