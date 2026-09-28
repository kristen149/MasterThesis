"""Execution-Accuracy (EX) evaluator.

Runs the predicted SQL and the gold SQL against the database and compares
the result sets. The comparison is order-sensitive when the gold contains
``ORDER BY`` (the user explicitly asked for an order), otherwise it's
multiset equality (different orderings of the same rows are equivalent).

Edge cases
----------
- Both queries fail to execute → score 0 (we don't reward shared brokenness).
- Predicted fails, gold succeeds → score 0.
- Predicted succeeds, gold fails → score 0 (gold is the source of truth;
  if it fails the question is unscorable, so we conservatively return 0).
- Different column count or column order is ignored — we compare row data
  only, casting each value to a normalized string so that 1 == "1" and
  1.0 == 1.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from ..executor import ExecResult, SQLExecutor
from .base import EvalResult, Evaluator


_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _normalize_cell(v) -> str:
    if v is None:
        return "<null>"
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.6f}"
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v).strip().lower()


def _normalize_row(row: tuple) -> tuple:
    return tuple(_normalize_cell(v) for v in row)


def _compare(pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool) -> bool:
    p = [_normalize_row(r) for r in pred_rows]
    g = [_normalize_row(r) for r in gold_rows]
    if ordered:
        return p == g
    return Counter(p) == Counter(g)


class ExecutionAccuracy(Evaluator):
    name = "EX"

    def __init__(self, executor: SQLExecutor) -> None:
        self.executor = executor

    def evaluate(
        self,
        predicted_sql: str,
        gold_sql: str,
        pred_exec: Optional[ExecResult] = None,
        gold_exec: Optional[ExecResult] = None,
    ) -> EvalResult:
        # Reuse precomputed results if the runner already ran them — saves
        # one execute call per sample in the common path.
        if pred_exec is None:
            pred_exec = self.executor.execute(predicted_sql)
        if gold_exec is None:
            gold_exec = self.executor.execute(gold_sql)

        if not pred_exec.ok or not gold_exec.ok:
            return EvalResult(
                score=0,
                detail={
                    "pred_ok": pred_exec.ok,
                    "gold_ok": gold_exec.ok,
                    "pred_error": pred_exec.error,
                    "gold_error": gold_exec.error,
                    "pred_rows_count": len(pred_exec.rows),
                    "gold_rows_count": len(gold_exec.rows),
                },
            )

        ordered = bool(_ORDER_BY.search(gold_sql or ""))
        match = _compare(pred_exec.rows, gold_exec.rows, ordered)
        return EvalResult(
            score=int(match),
            detail={
                "pred_ok": True,
                "gold_ok": True,
                "ordered": ordered,
                "pred_rows_count": len(pred_exec.rows),
                "gold_rows_count": len(gold_exec.rows),
                # Bound serialized size — first 5 rows is enough for debugging.
                "pred_rows_head": [list(r) for r in pred_exec.rows[:5]],
                "gold_rows_head": [list(r) for r in gold_exec.rows[:5]],
            },
        )
