"""Проверка архитектурных границ для ЛР1."""

from __future__ import annotations

import ast
from pathlib import Path

DOMAIN_DIR = Path(__file__).resolve().parents[1] / "domain"
FORBIDDEN_IMPORT_PREFIXES = ("PySide6", "matplotlib", "tkinter")


def _extract_import_roots(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_domain_layer_has_no_ui_or_plot_dependencies() -> None:
    offenders: list[str] = []
    for file_path in DOMAIN_DIR.glob("*.py"):
        imports = _extract_import_roots(file_path)
        if any(prefix in imports for prefix in FORBIDDEN_IMPORT_PREFIXES):
            offenders.append(file_path.name)

    assert not offenders, f"Нарушены границы доменного слоя: {offenders}"
