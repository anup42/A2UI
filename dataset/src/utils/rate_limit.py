import threading
import time


class RateLimiter:
    def __init__(self, qps: float, call_sleep_seconds: float = 0.0):
        self._lock = threading.Lock()
        self._qps = max(qps, 0.0)
        self._min_interval = 1.0 / self._qps if self._qps > 0 else 0.0
        self._last_time = 0.0
        self._call_sleep_seconds = max(call_sleep_seconds, 0.0)

    def acquire(self):
        if self._call_sleep_seconds > 0:
            time.sleep(self._call_sleep_seconds)
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            wait = self._min_interval - (now - self._last_time)
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self._last_time = now
