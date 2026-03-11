"""Основные модели данных доменного слоя.

Модели здесь играют роль контракта между слоями:
- UI собирает ввод и формирует `InputConfig`;
- прикладной слой возвращает `RunReport`;
- алгоритмы поиска отдают `SearchResult` и список `IterationRow`.

Почти все модели неизменяемые (`frozen=True`), чтобы результаты расчётов
не менялись случайно после построения отчёта.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from lr1.infrastructure.logging import configure_logging


configure_logging()
logger = logging.getLogger("lr1.app_models")


FunctionCallable = Callable[[float], float]


@dataclass(frozen=True)
class CoefficientSpec:
    """Описание одного редактируемого коэффициента функции в интерфейсе."""
    key: str
    label: str
    default: float


@dataclass(frozen=True)
class FunctionSpec:
    """Готовая функция для расчётов вместе с её аналитическими свойствами."""
    key: str
    title: str
    func: FunctionCallable
    forbidden_points: Tuple[float, ...] = ()
    stationary_points: Tuple[float, ...] = ()
    coefficient_values: Tuple[Tuple[str, float], ...] = ()
    formula_hint: str = ""

    def __post_init__(self) -> None:
        logger.debug(
            "FunctionSpec created key=%s title=%s forbidden_points=%s stationary_points=%s coefficients=%s",
            self.key,
            self.title,
            self.forbidden_points,
            self.stationary_points,
            self.coefficient_values,
        )


@dataclass(frozen=True)
class FunctionTemplateSpec:
    """Шаблон функции, из которого UI и сервисы собирают `FunctionSpec`."""
    key: str
    title: str
    formula_hint: str
    coefficients: Tuple[CoefficientSpec, ...]
    builder: Callable[[Dict[str, float]], FunctionSpec]


@dataclass(frozen=True)
class IterationRow:
    """Одна строка итерационной таблицы метода поиска."""
    k: int
    a: float
    b: float
    lam: float
    mu: float
    f_lam: float
    f_mu: float


@dataclass(frozen=True)
class SearchResult:
    """Результат работы одного конкретного метода на одном наборе параметров."""
    method: str
    kind: str
    x_opt: float
    f_opt: float
    iterations: List[IterationRow]
    func_evals: int
    interval_initial: Tuple[float, float]
    interval_final: Tuple[float, float]

    def __post_init__(self) -> None:
        logger.debug(
            "SearchResult created method=%s kind=%s x_opt=%.10f f_opt=%.10f iterations=%d evals=%d interval_final=%s",
            self.method,
            self.kind,
            self.x_opt,
            self.f_opt,
            len(self.iterations),
            self.func_evals,
            self.interval_final,
        )


@dataclass(frozen=True)
class MethodSpec:
    """Метаданные метода поиска и ссылка на его реализацию."""
    key: str
    title: str
    runner: Callable[[FunctionCallable, float, float, float, float, str], SearchResult]

    def __post_init__(self) -> None:
        logger.debug("MethodSpec created key=%s title=%s runner=%s", self.key, self.title, self.runner.__name__)


@dataclass(frozen=True)
class InputConfig:
    """Проверенный и типизированный набор входных параметров расчёта."""
    function_spec: FunctionSpec
    kind: str
    method_key: str
    interval_raw: Tuple[float, float]
    interval: Tuple[float, float]
    eps: float
    l: float

    def __post_init__(self) -> None:
        logger.debug(
            "InputConfig created function=%s kind=%s method=%s interval_raw=%s interval=%s eps=%s l=%s",
            self.function_spec.key,
            self.kind,
            self.method_key,
            self.interval_raw,
            self.interval,
            self.eps,
            self.l,
        )


@dataclass(frozen=True)
class GridRunResult:
    """Один успешный прогон внутри серии расчётов по сетке параметров."""
    method_key: str
    eps: float
    l: float
    result: SearchResult

    def __post_init__(self) -> None:
        logger.debug(
            "GridRunResult created method=%s eps=%s l=%s evals=%d x_opt=%.10f",
            self.method_key,
            self.eps,
            self.l,
            self.result.func_evals,
            self.result.x_opt,
        )


@dataclass(frozen=True)
class SkippedRun:
    """Описание набора параметров, который был пропущен как невалидный."""
    method_key: str
    eps: float
    l: float
    reason: str

    def __post_init__(self) -> None:
        logger.debug(
            "SkippedRun created method=%s eps=%s l=%s reason=%s",
            self.method_key,
            self.eps,
            self.l,
            self.reason,
        )


@dataclass(frozen=True)
class ReferencePoint:
    """Теоретический ориентир для сравнения численного результата."""
    x: float
    f: float
    source: str


@dataclass(frozen=True)
class RunReport:
    """Итоговый отчёт для отображения в UI.

    Это не "сырой" результат одного метода, а агрегат:
    - для одиночного режима содержит все выполненные методы;
    - для серии расчётов содержит лучшие результаты и список прогонов;
    - хранит всё, что нужно вкладкам `Сводка`, `Итерации` и `График`.
    """
    results_by_method: Dict[str, SearchResult]
    method_keys: Tuple[str, ...]
    default_method_key: Optional[str]
    requested_method_key: str
    function_spec: FunctionSpec
    kind: str
    interval_raw: Tuple[float, float]
    interval: Tuple[float, float]
    eps: Optional[float]
    l: Optional[float]
    plot_range: Tuple[float, float]
    reference_point: Optional[ReferencePoint] = None
    mode: str = "single"
    grid_runs_by_method: Dict[str, Tuple[GridRunResult, ...]] = field(default_factory=dict)
    skipped_runs: Tuple[SkippedRun, ...] = ()

    def __post_init__(self) -> None:
        logger.debug(
            "RunReport created function=%s kind=%s mode=%s methods=%s requested_method=%s default_method=%s results=%d grid_methods=%d skipped=%d interval_raw=%s interval=%s has_reference=%s",
            self.function_spec.key,
            self.kind,
            self.mode,
            self.method_keys,
            self.requested_method_key,
            self.default_method_key,
            len(self.results_by_method),
            len(self.grid_runs_by_method),
            len(self.skipped_runs),
            self.interval_raw,
            self.interval,
            self.reference_point is not None,
        )


@dataclass
class AppState:
    """Минимальное состояние GUI между действиями пользователя."""
    last_report: Optional[RunReport] = None
    busy: bool = False
    selected_table_method: str = ""
    selected_grid_run_index: int = 0

    def __post_init__(self) -> None:
        logger.debug(
            "AppState created busy=%s selected_table_method=%s selected_grid_run_index=%d",
            self.busy,
            self.selected_table_method,
            self.selected_grid_run_index,
        )
