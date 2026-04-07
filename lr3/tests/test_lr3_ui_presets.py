"""UI-smoke тесты preset-контракта ЛР3."""

from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication, QLabel

from lr3.domain.expression import analyze_local_extremum
from lr3.ui.window import GradientMethodsWindow


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
