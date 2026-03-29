"""Централизованные настройки лабораторной работы.

Здесь хранятся:
- значения по умолчанию для полей интерфейса;
- численные допуски, которые используются во внутренних вычислениях.
"""

DEFAULT_INTERVAL = ("-10", "10")
DEFAULT_INPUT_EPS = "0.01"
DEFAULT_L = "0.1"

POLYNOMIAL_TOLERANCE = 1e-12
ROOT_TOLERANCE = 1e-9
DENOMINATOR_TOLERANCE = 1e-15
FORBIDDEN_POINT_TOLERANCE = 1e-4
INTERVAL_SHIFT_BASE = 1e-8
MIN_INTERVAL_SHIFT = 1e-10
