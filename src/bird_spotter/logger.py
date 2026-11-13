"""Thread-safe live ring-buffer logger for the British Bird Spotter dashboard and service."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import threading
import time
from typing import Any

LEVEL_VALUES: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARN": 30,
    "WARNING": 30,
    "ERROR": 40,
}


class SpotterLogger:
    """In-memory ring buffer logger with level control and per-key rate limiting for 30 FPS loops."""

    def __init__(self, max_entries: int = 400, default_level: str = "INFO") -> None:
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._seq: int = 0
        self._level: str = default_level.upper() if default_level.upper() in ("DEBUG", "INFO") else "INFO"
        self._throttle_ts: dict[str, float] = {}
        self.echo_stdout: bool = True

    @property
    def level(self) -> str:
        with self._lock:
            return self._level

    def set_level(self, level: str) -> str:
        lvl = str(level).upper().strip()
        if lvl not in ("DEBUG", "INFO", "WARN", "ERROR"):
            lvl = "INFO"
        with self._lock:
             prev = self._level
             self._level = lvl
        if prev != lvl:
            self.info("SYSTEM", f"Live log level changed: {prev} -> {lvl}")
        return lvl

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._throttle_ts.clear()

    def log(self, level: str, category: str, msg: str) -> dict[str, Any] | None:
        lvl = level.upper()
        if lvl == "WARNING":
            lvl = "WARN"
        now_epoch = time.time()
        dt = datetime.fromtimestamp(now_epoch)
        ts_str = dt.strftime("%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}"

        with self._lock:
            self._seq += 1
            entry = {
                "seq": self._seq,
                "ts": ts_str,
                "epoch": round(now_epoch, 3),
                "level": lvl,
                "category": category.upper(),
                "msg": str(msg),
            }
            self._entries.append(entry)
            active_lvl_num = LEVEL_VALUES.get(self._level, 20)

        entry_lvl_num = LEVEL_VALUES.get(lvl, 20)
        if self.echo_stdout and entry_lvl_num >= active_lvl_num:
            print(f"[{ts_str}] [{lvl:<5}] [{category.upper():<10}] {msg}", flush=True)

        return entry

    def debug(self, category: str, msg: str) -> dict[str, Any] | None:
        return self.log("DEBUG", category, msg)

    def debug_throttled(
        self,
        key: str,
        category: str,
        msg: str,
        interval_sec: float = 0.40,
    ) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            last = self._throttle_ts.get(key, 0.0)
            if (now - last) < interval_sec:
                return None
            self._throttle_ts[key] = now
        return self.log("DEBUG", category, msg)

    def info(self, category: str, msg: str) -> dict[str, Any] | None:
        return self.log("INFO", category, msg)

    def warn(self, category: str, msg: str) -> dict[str, Any] | None:
        return self.log("WARN", category, msg)

    def error(self, category: str, msg: str) -> dict[str, Any] | None:
        return self.log("ERROR", category, msg)

    def get_entries(
        self,
        min_level: str | None = None,
        since_seq: int = 0,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        with self._lock:
            lvl_name = (min_level or self._level).upper()
            min_num = LEVEL_VALUES.get(lvl_name, 20)
            filtered = [
                e
                for e in self._entries
                if e["seq"] > since_seq and LEVEL_VALUES.get(e["level"], 20) >= min_num
            ]
            if limit > 0 and len(filtered) > limit:
                return filtered[-limit:]
            return filtered


# Global shared singleton logger
spotter_log = SpotterLogger()
