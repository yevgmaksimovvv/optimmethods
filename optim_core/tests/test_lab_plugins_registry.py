"""Тест контракта статического shell для лабораторных работ."""

from __future__ import annotations

from PySide6.QtCore import Qt

from optim_core.labs.shell import APP_TITLE, DEFAULT_LABS, LabsShellWindow


def test_shell_contains_expected_labs(qapp_offscreen) -> None:
    app = qapp_offscreen
    window = LabsShellWindow()
    app.processEvents()

    assert window.windowTitle() == APP_TITLE
    assert window.nav_list.count() == len(DEFAULT_LABS)
    assert [window.nav_list.item(index).data(Qt.UserRole) for index in range(window.nav_list.count())] == [
        lab.lab_id for lab in DEFAULT_LABS
    ]
    assert [window.nav_list.item(index).text() for index in range(window.nav_list.count())] == [
        f"{lab.title} — {lab.description}" for lab in DEFAULT_LABS
    ]
    assert window.open_external_button.isEnabled() is True
    window.close()
