"""Общие проверки архитектурных границ слоев."""

from __future__ import annotations

import ast
from pathlib import Path


def collect_import_roots(module_path: Path) -> set[str]:
    """Возвращает набор корневых импортов из python-модуля."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def find_forbidden_import_offenders(layer_dir: Path, forbidden_import_prefixes: tuple[str, ...]) -> list[str]:
    """Ищет файлы слоя, которые импортируют запрещенные пакеты."""
    offenders: list[str] = []
    for file_path in layer_dir.glob("*.py"):
        imports = collect_import_roots(file_path)
        if any(prefix in imports for prefix in forbidden_import_prefixes):
            offenders.append(file_path.name)
    return offenders
