import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

if __package__:
    from .logging_setup import configure_logging
else:
    from logging_setup import configure_logging


configure_logging()
logger = logging.getLogger("lr1.app_models")


FunctionCallable = Callable[[float], float]


@dataclass(frozen=True)
class FunctionSpec:
    key: str
    title: str
    func: FunctionCallable
    forbidden_points: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        logger.debug(
            "FunctionSpec created key=%s title=%s forbidden_points=%s",
            self.key,
            self.title,
            self.forbidden_points,
        )


@dataclass(frozen=True)
class IterationRow:
    k: int
    a: float
    b: float
    lam: float
    mu: float
    f_lam: float
    f_mu: float


@dataclass(frozen=True)
class SearchResult:
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
    key: str
    title: str
    runner: Callable[[FunctionCallable, float, float, float, float, str], SearchResult]

    def __post_init__(self) -> None:
        logger.debug("MethodSpec created key=%s title=%s runner=%s", self.key, self.title, self.runner.__name__)


@dataclass(frozen=True)
class InputConfig:
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
class RunReport:
    summary_text: str
    results_by_method: Dict[str, SearchResult]
    method_keys: Tuple[str, ...]
    default_method_key: Optional[str]
    function_spec: FunctionSpec
    kind: str
    interval_raw: Tuple[float, float]
    interval: Tuple[float, float]
    eps: Optional[float]
    l: Optional[float]
    plot_range: Tuple[float, float]
    analytic_note: str = ""
    reference_x: Optional[float] = None
    reference_f: Optional[float] = None
    reference_source: str = ""
    observations: Tuple[str, ...] = ()
    mode: str = "single"
    grid_runs_by_method: Dict[str, Tuple[GridRunResult, ...]] = field(default_factory=dict)
    skipped_runs: Tuple[SkippedRun, ...] = ()

    def __post_init__(self) -> None:
        logger.debug(
            "RunReport created function=%s kind=%s mode=%s methods=%s default_method=%s results=%d grid_methods=%d skipped=%d interval_raw=%s interval=%s summary_len=%d",
            self.function_spec.key,
            self.kind,
            self.mode,
            self.method_keys,
            self.default_method_key,
            len(self.results_by_method),
            len(self.grid_runs_by_method),
            len(self.skipped_runs),
            self.interval_raw,
            self.interval,
            len(self.summary_text),
        )


@dataclass
class AppState:
    last_report: Optional[RunReport] = None
    busy: bool = False
    pending_plot_refresh: bool = False
    selected_table_method: str = ""
    selected_grid_run_index: int = 0

    def __post_init__(self) -> None:
        logger.debug(
            "AppState created busy=%s selected_table_method=%s selected_grid_run_index=%d",
            self.busy,
            self.selected_table_method,
            self.selected_grid_run_index,
        )
