import logging

if __package__:
    from .app_models import FunctionSpec
    from .logging_setup import configure_logging
else:
    from app_models import FunctionSpec
    from logging_setup import configure_logging


configure_logging()
logger = logging.getLogger("lr1.function_defs")


def f1(x: float) -> float:
    return 10.0 * x - 2.0 * x * x + 3.0


def f2(x: float) -> float:
    den = x * x + 2.0 * x - 8.0
    if abs(den) < 1e-15:
        logger.warning("f2 undefined at x=%s due to near-zero denominator=%s", x, den)
        raise ZeroDivisionError(f"f2 is undefined at x={x}")
    return (2.0 * x * x + 3.0) / den


FUNCTION_SPECS = {
    "F1": FunctionSpec(
        key="F1",
        title="F1(x)=10x-2x^2+3",
        func=f1,
        forbidden_points=(),
    ),
    "F2": FunctionSpec(
        key="F2",
        title="F2(x)=(2x^2+3)/(x^2+2x-8)",
        func=f2,
        forbidden_points=(-4.0, 2.0),
    ),
}

logger.debug("Function definitions loaded. available_functions=%s", tuple(FUNCTION_SPECS))
