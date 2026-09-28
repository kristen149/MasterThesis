"""SQL executor.

Executes a SQL string against the SQLite event_log DB in read-only mode,
with a per-query timeout. Exceptions become structured ``ExecResult``
records — they never escape this module.

Cross-platform timeout
----------------------
``signal.alarm`` is POSIX-only. We use SQLite's own progress handler
(via ``Connection.set_progress_handler``) plus a wall-clock check. This
works on Linux, macOS, and Windows.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple]
    columns: list[str]
    error: Optional[str] = None
    elapsed_ms: float = 0.0


class SQLExecutor:
    def __init__(self, db_path: str, timeout_seconds: int = 5) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.timeout_seconds = timeout_seconds
        # Thread-local connections so the executor is safe to share if/when
        # we add concurrency. SQLite connections aren't thread-safe by default.
        self._tls = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._tls.conn = conn
        return conn

    def execute(self, sql: str) -> ExecResult:
        if not sql or not sql.strip():
            return ExecResult(ok=False, rows=[], columns=[], error="empty SQL")

        start = time.perf_counter()
        deadline = start + self.timeout_seconds

        # Progress handler is called every N VM instructions. Returning non-zero
        # cancels the in-flight statement. This gives us a portable timeout.
        def _progress() -> int:
            return 1 if time.perf_counter() > deadline else 0

        conn = self._conn()
        conn.set_progress_handler(_progress, 1000)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
            return ExecResult(
                ok=True,
                rows=rows,
                columns=cols,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
        except sqlite3.OperationalError as e:
            msg = str(e)
            # The progress handler raises OperationalError("interrupted").
            if "interrupted" in msg.lower():
                msg = f"query timeout after {self.timeout_seconds}s"
            return ExecResult(
                ok=False,
                rows=[],
                columns=[],
                error=msg,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
        except sqlite3.Error as e:
            return ExecResult(
                ok=False,
                rows=[],
                columns=[],
                error=f"{type(e).__name__}: {e}",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
        finally:
            conn.set_progress_handler(None, 0)

    def close(self) -> None:
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None
