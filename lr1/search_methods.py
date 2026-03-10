import logging
import math
import time
from typing import Dict, List, Optional, Tuple

if __package__:
    from .app_models import IterationRow, MethodSpec, SearchResult
    from .logging_setup import configure_logging
else:
    from app_models import IterationRow, MethodSpec, SearchResult
    from logging_setup import configure_logging


configure_logging()
logger = logging.getLogger("lr1.search_methods")


def better(v1: float, v2: float, kind: str) -> bool:
    logger.debug("better compare kind=%s v1=%s v2=%s", kind, v1, v2)
    if kind == "max":
        return v1 > v2
    if kind == "min":
        return v1 < v2
    raise ValueError("Тип поиска должен быть 'max' или 'min'")


def compare_left_right(left_val: float, right_val: float, kind: str) -> str:
    return "right" if better(left_val, right_val, kind) else "left"


def sanitize_interval(
    a: float,
    b: float,
    forbidden_points: Optional[List[float]] = None,
    shift: float = 1e-8,
) -> Tuple[float, float]:
    logger.debug(
        "sanitize_interval start a=%s b=%s forbidden_points=%s shift=%s",
        a,
        b,
        forbidden_points,
        shift,
    )
    if a >= b:
        raise ValueError(f"Invalid interval [{a}, {b}]")

    forbidden_points = forbidden_points or []
    for point in forbidden_points:
        if a < point < b:
            raise ValueError(
                f"Интервал [{a}, {b}] содержит точку разрыва x={point}. "
                "Выбирай интервал только по одну сторону от разрыва."
            )
        if abs(a - point) < 1e-15:
            a += shift
        if abs(b - point) < 1e-15:
            b -= shift

    if a >= b:
        raise ValueError("Интервал схлопнулся после сдвига границ.")

    logger.debug("sanitize_interval result a=%s b=%s", a, b)
    return a, b
def dichotomy_search(
    func,
    a: float,
    b: float,
    eps: float,
    l: float,
    kind: str = "max",
) -> SearchResult:
    started = time.perf_counter()
    logger.info("dichotomy_search start a=%s b=%s eps=%s l=%s kind=%s", a, b, eps, l, kind)
    if eps <= 0:
        raise ValueError("Параметр ε должен быть больше 0")
    if l <= 0:
        raise ValueError("Параметр l должен быть больше 0")
    if eps >= l:
        raise ValueError("Для метода дихотомии нужно ε < l")

    initial = (a, b)
    rows: List[IterationRow] = []
    evals = 0
    k = 1

    while (b - a) > l:
        mid = (a + b) / 2.0
        lam = mid - eps
        mu = mid + eps
        f_lam = func(lam)
        f_mu = func(mu)
        evals += 2

        rows.append(IterationRow(k, a, b, lam, mu, f_lam, f_mu))
        logger.debug(
            "dichotomy iteration=%d interval=(%.10f, %.10f) lam=%.10f mu=%.10f f_lam=%.10f f_mu=%.10f",
            k,
            a,
            b,
            lam,
            mu,
            f_lam,
            f_mu,
        )

        side = compare_left_right(f_lam, f_mu, kind)
        if side == "right":
            a = lam
        else:
            b = mu
        k += 1

    x_opt = (a + b) / 2.0
    f_opt = func(x_opt)
    evals += 1

    result = SearchResult(
        method="Дихотомия",
        kind=kind,
        x_opt=x_opt,
        f_opt=f_opt,
        iterations=rows,
        func_evals=evals,
        interval_initial=initial,
        interval_final=(a, b),
    )
    logger.info(
        "dichotomy_search done iterations=%d evals=%d x_opt=%.10f f_opt=%.10f duration_ms=%.2f",
        len(rows),
        evals,
        x_opt,
        f_opt,
        (time.perf_counter() - started) * 1000.0,
    )
    return result


def golden_section_search(
    func,
    a: float,
    b: float,
    eps: float,
    l: float,
    kind: str = "max",
) -> SearchResult:
    started = time.perf_counter()
    logger.info("golden_section_search start a=%s b=%s eps=%s l=%s kind=%s", a, b, eps, l, kind)
    del eps

    if l <= 0:
        raise ValueError("Параметр l должен быть больше 0")

    initial = (a, b)
    rows: List[IterationRow] = []
    evals = 0
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    lam = b - (b - a) / phi
    mu = a + (b - a) / phi
    f_lam = func(lam)
    f_mu = func(mu)
    evals += 2
    k = 1

    while (b - a) > l:
        rows.append(IterationRow(k, a, b, lam, mu, f_lam, f_mu))
        logger.debug(
            "golden iteration=%d interval=(%.10f, %.10f) lam=%.10f mu=%.10f f_lam=%.10f f_mu=%.10f",
            k,
            a,
            b,
            lam,
            mu,
            f_lam,
            f_mu,
        )

        side = compare_left_right(f_lam, f_mu, kind)
        if side == "right":
            a = lam
            lam = mu
            f_lam = f_mu
            mu = a + (b - a) / phi
            f_mu = func(mu)
            evals += 1
        else:
            b = mu
            mu = lam
            f_mu = f_lam
            lam = b - (b - a) / phi
            f_lam = func(lam)
            evals += 1
        k += 1

    x_opt = (a + b) / 2.0
    f_opt = func(x_opt)
    evals += 1

    result = SearchResult(
        method="Золотое сечение",
        kind=kind,
        x_opt=x_opt,
        f_opt=f_opt,
        iterations=rows,
        func_evals=evals,
        interval_initial=initial,
        interval_final=(a, b),
    )
    logger.info(
        "golden_section_search done iterations=%d evals=%d x_opt=%.10f f_opt=%.10f duration_ms=%.2f",
        len(rows),
        evals,
        x_opt,
        f_opt,
        (time.perf_counter() - started) * 1000.0,
    )
    return result


def fibonacci_numbers_until(limit: float) -> List[int]:
    logger.debug("fibonacci_numbers_until limit=%s", limit)
    fib = [0, 1, 1]
    while fib[-1] < limit:
        fib.append(fib[-1] + fib[-2])
    logger.debug("fibonacci_numbers_until generated n=%d last=%s", len(fib), fib[-1])
    return fib


def fibonacci_search(
    func,
    a: float,
    b: float,
    eps: float,
    l: float,
    kind: str = "max",
) -> SearchResult:
    started = time.perf_counter()
    logger.info("fibonacci_search start a=%s b=%s eps=%s l=%s kind=%s", a, b, eps, l, kind)
    if l <= 0:
        raise ValueError("Параметр l должен быть больше 0")
    if eps <= 0:
        raise ValueError("Параметр ε должен быть больше 0")

    initial = (a, b)
    rows: List[IterationRow] = []
    evals = 0

    length = b - a
    fib = fibonacci_numbers_until(length / l)
    n = len(fib) - 1

    if n < 3:
        x_opt = (a + b) / 2.0
        f_opt = func(x_opt)
        result = SearchResult(
            method="Фибоначчи",
            kind=kind,
            x_opt=x_opt,
            f_opt=f_opt,
            iterations=[],
            func_evals=1,
            interval_initial=initial,
            interval_final=(a, b),
        )
        logger.info(
            "fibonacci_search short-circuit x_opt=%.10f f_opt=%.10f duration_ms=%.2f",
            x_opt,
            f_opt,
            (time.perf_counter() - started) * 1000.0,
        )
        return result

    lam = a + (fib[n - 2] / fib[n]) * (b - a)
    mu = a + (fib[n - 1] / fib[n]) * (b - a)
    f_lam = func(lam)
    f_mu = func(mu)
    evals += 2

    k = 1
    for i in range(1, n - 2):
        rows.append(IterationRow(k, a, b, lam, mu, f_lam, f_mu))
        logger.debug(
            "fibonacci iteration=%d interval=(%.10f, %.10f) lam=%.10f mu=%.10f f_lam=%.10f f_mu=%.10f",
            k,
            a,
            b,
            lam,
            mu,
            f_lam,
            f_mu,
        )

        side = compare_left_right(f_lam, f_mu, kind)
        if side == "right":
            a = lam
            lam = mu
            f_lam = f_mu
            mu = a + (fib[n - i - 1] / fib[n - i]) * (b - a)
            f_mu = func(mu)
            evals += 1
        else:
            b = mu
            mu = lam
            f_mu = f_lam
            lam = a + (fib[n - i - 2] / fib[n - i]) * (b - a)
            f_lam = func(lam)
            evals += 1
        k += 1

    rows.append(IterationRow(k, a, b, lam, mu, f_lam, f_mu))
    lambda_n = lam
    mu_n = lambda_n + eps

    f_mu_n = func(mu_n)
    evals += 1

    if compare_left_right(f_lam, f_mu_n, kind) == "right":
        a_final = lambda_n
        b_final = b
    else:
        a_final = a
        b_final = lambda_n

    x_opt = (a_final + b_final) / 2.0
    f_opt = func(x_opt)
    evals += 1

    result = SearchResult(
        method="Фибоначчи",
        kind=kind,
        x_opt=x_opt,
        f_opt=f_opt,
        iterations=rows,
        func_evals=evals,
        interval_initial=initial,
        interval_final=(a_final, b_final),
    )
    logger.info(
        "fibonacci_search done iterations=%d evals=%d x_opt=%.10f f_opt=%.10f duration_ms=%.2f",
        len(rows),
        evals,
        x_opt,
        f_opt,
        (time.perf_counter() - started) * 1000.0,
    )
    return result


METHOD_ORDER = ("dichotomy", "golden", "fibonacci")

METHOD_SPECS: Dict[str, MethodSpec] = {
    "dichotomy": MethodSpec(
        key="dichotomy",
        title="Дихотомия",
        runner=dichotomy_search,
    ),
    "golden": MethodSpec(
        key="golden",
        title="Золотое сечение",
        runner=golden_section_search,
    ),
    "fibonacci": MethodSpec(
        key="fibonacci",
        title="Фибоначчи",
        runner=fibonacci_search,
    ),
}

logger.debug("Search methods registered order=%s", METHOD_ORDER)
