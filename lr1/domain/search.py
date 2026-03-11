"""Реализации методов одномерного поиска экстремума.

В модуле собраны три алгоритма:
- дихотомия;
- золотое сечение;
- Фибоначчи.
"""

import logging
import math
import time
from typing import Dict, List, Tuple

from lr1.domain.models import IterationRow, MethodSpec, SearchResult
from lr1.domain.numerical import choose_side, scaled_interval_shift
from lr1.infrastructure.logging import configure_logging


configure_logging()
logger = logging.getLogger("lr1.search_methods")


def _safe_eval(
    func,
    x: float,
    a: float,
    b: float,
    max_retries: int = 16,
) -> Tuple[float, int]:
    """Вычисляет `func(x)`, а при разрыве пробует соседние точки внутри интервала."""
    interval_span = b - a
    base_shift = max(1e-12, min(scaled_interval_shift(a, b), interval_span / 16.0))
    attempts = 1
    try:
        return func(x), attempts
    except ZeroDivisionError:
        logger.debug(
            "safe_eval undefined at x=%.12f in interval=(%.12f, %.12f), trying neighbours with base_shift=%s",
            x,
            a,
            b,
            base_shift,
        )

    for retry in range(max_retries):
        delta = base_shift * (2.0 ** retry)
        left = x - delta
        right = x + delta
        candidates = []
        if a < left < b:
            candidates.append(left)
        if a < right < b:
            candidates.append(right)

        for candidate in candidates:
            attempts += 1
            try:
                value = func(candidate)
                logger.debug(
                    "safe_eval fallback used x=%.12f for requested x=%.12f after %d attempts",
                    candidate,
                    x,
                    attempts,
                )
                return value, attempts
            except ZeroDivisionError:
                continue

    raise ValueError(
        f"Не удалось вычислить функцию около x={x:.12g} на интервале [{a:.12g}, {b:.12g}] "
        "из-за точки разрыва."
    )


def dichotomy_search(
    func,
    a: float,
    b: float,
    eps: float,
    l: float,
    kind: str = "max",
) -> SearchResult:
    """Реализация метода дихотомии.

    Идея метода:
    - на каждом шаге берётся середина текущего интервала;
    - рядом с серединой выбираются две точки `λ = mid - ε` и `μ = mid + ε`;
    - по значениям функции в этих точках выбирается половина интервала,
      которая содержит лучший экстремум.
    """
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
        f_lam, used = _safe_eval(func, lam, a, b)
        evals += used
        f_mu, used = _safe_eval(func, mu, a, b)
        evals += used

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

        side = choose_side(f_lam, f_mu, kind)
        if side == "right":
            a = lam
        else:
            b = mu
        k += 1

    x_opt = (a + b) / 2.0
    f_opt, used = _safe_eval(func, x_opt, a, b)
    evals += used

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
    """Реализация метода золотого сечения в классическом виде.

    Метод использует фиксированное отношение точек внутри интервала,
    благодаря чему на каждом шаге можно переиспользовать одну из старых
    пробных точек и вычислять только одно новое значение функции.
    """
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
    f_lam, used = _safe_eval(func, lam, a, b)
    evals += used
    f_mu, used = _safe_eval(func, mu, a, b)
    evals += used
    k = 1
    rows.append(IterationRow(k, a, b, lam, mu, f_lam, f_mu))
    while (b - a) > l:
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

        side = choose_side(f_lam, f_mu, kind)
        if side == "right":
            a = lam
            lam = mu
            f_lam = f_mu
            mu = a + (b - a) / phi
            f_mu, used = _safe_eval(func, mu, a, b)
            evals += used
        else:
            b = mu
            mu = lam
            f_mu = f_lam
            lam = b - (b - a) / phi
            f_lam, used = _safe_eval(func, lam, a, b)
            evals += used
        k += 1
        rows.append(IterationRow(k, a, b, lam, mu, f_lam, f_mu))

    x_opt = (a + b) / 2.0
    f_opt, used = _safe_eval(func, x_opt, a, b)
    evals += used

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
    """Генерирует числа Фибоначчи до первого значения, не меньшего `limit`."""
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
    """Реализация метода Фибоначчи.

    Метод похож на золотое сечение, но длины шагов задаются отношениями
    соседних чисел Фибоначчи. Это позволяет заранее оценить число итераций
    для заданной длины конечного интервала `l`.
    """
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
        f_opt, used = _safe_eval(func, x_opt, a, b)
        result = SearchResult(
            method="Фибоначчи",
            kind=kind,
            x_opt=x_opt,
            f_opt=f_opt,
            iterations=[],
            func_evals=used,
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
    f_lam, used = _safe_eval(func, lam, a, b)
    evals += used
    f_mu, used = _safe_eval(func, mu, a, b)
    evals += used
    k = 1
    rows.append(IterationRow(k, a, b, lam, mu, f_lam, f_mu))
    for i in range(1, n - 2):
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

        side = choose_side(f_lam, f_mu, kind)
        if side == "right":
            a = lam
            lam = mu
            f_lam = f_mu
            mu = a + (fib[n - i - 1] / fib[n - i]) * (b - a)
            f_mu, used = _safe_eval(func, mu, a, b)
            evals += used
        else:
            b = mu
            mu = lam
            f_mu = f_lam
            lam = a + (fib[n - i - 2] / fib[n - i]) * (b - a)
            f_lam, used = _safe_eval(func, lam, a, b)
            evals += used
        k += 1
        rows.append(IterationRow(k, a, b, lam, mu, f_lam, f_mu))

    lambda_n = lam
    mu_n = lambda_n + eps

    f_mu_n, used = _safe_eval(func, mu_n, a, b)
    evals += used
    rows.append(IterationRow(k + 1, a, b, lambda_n, mu_n, f_lam, f_mu_n))

    if choose_side(f_lam, f_mu_n, kind) == "right":
        a_final = lambda_n
        b_final = b
    else:
        a_final = a
        b_final = lambda_n

    x_opt = (a_final + b_final) / 2.0
    f_opt, used = _safe_eval(func, x_opt, a_final, b_final)
    evals += used
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
