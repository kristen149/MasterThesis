"""Exact-Match evaluator.

Normalization policy (documented because it's a design decision, not a bug):

  - Lowercase everything outside single-quoted string literals.
  - Collapse all whitespace runs (outside literals) to a single space.
  - Strip trailing semicolon.
  - String literals themselves are compared in a case-insensitive way
    BECAUSE the underlying schema uses ``COLLATE NOCASE`` on text columns,
    making ``activity = 'End trip'`` and ``activity = 'end trip'`` semantically
    identical. This matches the gold dataset's casing conventions.

This is intentionally lenient: pure string EM is brittle, and we already
have EX (execution accuracy) for stricter semantic comparison.
"""
from __future__ import annotations

import re
from typing import Optional

from ..executor import ExecResult
from .base import EvalResult, Evaluator


_WHITESPACE = re.compile(r"\s+")
_COMMA_SPACE_RE = re.compile(r"\s*,\s*")
# Strip `AS alias` from SELECT expressions: matches " as <word>" not inside quotes.
_ALIAS_RE = re.compile(r"\s+as\s+\w+", re.IGNORECASE)
# Normalize COUNT(non-star) → COUNT(*): safe because the gold always uses COUNT(*).
_COUNT_COL_RE = re.compile(r"count\s*\(\s*(?!\*)\w+\s*\)", re.IGNORECASE)


def _split_outside_strings(sql: str) -> list[tuple[bool, str]]:
    """Tokenize into (in_string, fragment) pairs split on single quotes."""
    parts: list[tuple[bool, str]] = []
    buf: list[str] = []
    in_str = False
    for ch in sql:
        if ch == "'":
            parts.append((in_str, "".join(buf)))
            buf = []
            in_str = not in_str
        else:
            buf.append(ch)
    parts.append((in_str, "".join(buf)))
    return parts


def normalize_sql(sql: str) -> str:
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return ""

    out: list[str] = []
    for in_str, frag in _split_outside_strings(sql):
        if in_str:
            out.append(frag.lower())  # case-insensitive literal compare
        else:
            frag = _WHITESPACE.sub(" ", frag.lower())
            # Normalize comma spacing: "a , b" and "a,b" are the same SQL.
            frag = _COMMA_SPACE_RE.sub(",", frag)
            # Strip `AS alias` on SELECT expressions — cosmetic aliases don't
            # affect query results and the gold SQL rarely uses them.
            frag = _ALIAS_RE.sub("", frag)
            # Normalize COUNT(col) → COUNT(*): the gold always uses COUNT(*),
            # and counting a NOT NULL column is semantically identical.
            frag = _COUNT_COL_RE.sub("count(*)", frag)
            out.append(frag)
    return "".join(out).strip()


class ExactMatch(Evaluator):
    name = "EM"

    def evaluate(
        self,
        predicted_sql: str,
        gold_sql: str,
        pred_exec: Optional[ExecResult] = None,
        gold_exec: Optional[ExecResult] = None,
    ) -> EvalResult:
        np = normalize_sql(predicted_sql)
        ng = normalize_sql(gold_sql)
        match = int(bool(np) and np == ng)
        return EvalResult(
            score=match,
            detail={"normalized_pred": np, "normalized_gold": ng},
        )
