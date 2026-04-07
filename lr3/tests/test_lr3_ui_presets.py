"""UI-smoke тесты preset-контракта ЛР3."""

from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication

from lr3.application.services import DEFAULT_CONJUGATE_EXPRESSION, DEFAULT_GRADIENT_EXPRESSION
from lr3.ui.window import GradientMethodsWindow


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_lr3_presets_are_fixed_and_custom_is_editable() -> None:
    _app()
    window = GradientMethodsWindow()

    assert set(window.preset_buttons) == {"gradient", "conjugate", "custom"}
    assert window.expression_input.isReadOnly()
    assert window.expression_input.text() == DEFAULT_GRADIENT_EXPRESSION

    window.preset_buttons["conjugate"].click()
    assert window.expression_input.isReadOnly()
    assert window.expression_input.text() == DEFAULT_CONJUGATE_EXPRESSION
    assert window.start_x1_input.text() == "1"
    assert window.start_x2_input.text() == "1"

    window.preset_buttons["custom"].click()
    assert not window.expression_input.isReadOnly()
    window.expression_input.setText("x1 + x2")
    assert window.expression_input.text() == "x1 + x2"


def test_lr3_method_buttons_update_numeric_defaults() -> None:
    _app()
    window = GradientMethodsWindow()

    window.method_buttons["conjugate"].click()

    assert window.start_x1_input.text() == "1"
    assert window.start_x2_input.text() == "1"
    assert window.initial_step_input.text() == "0.2"
    assert window.max_iterations_input.text() == "300"

