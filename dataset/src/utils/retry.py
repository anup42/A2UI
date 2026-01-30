import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> T:
    attempt = 1
    while True:
        try:
            return fn()
        except retry_on as exc:
            if attempt >= max_attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if hasattr(exc, "code") and getattr(exc, "code") == 429:
                delay = max(delay, 30.0 * attempt)
            time.sleep(delay)
            attempt += 1
