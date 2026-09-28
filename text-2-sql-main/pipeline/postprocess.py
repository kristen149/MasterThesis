"""Stateless SQL postprocessor.

Models routinely add markdown fences, prose, or extra statements around the
SQL we want. This module turns raw model output into a single executable
SQL string. Pure functions — no I/O, no model knowledge, no DB access.

Order of operations:
  1. Strip ChatML special tokens (<|im_end|> etc.) — Qwen local-model artefact.
  2. Strip markdown fences (```sql ... ``` or ``` ... ```).
  3. If CoT was used, extract the line after "Final SQL:".
  4. Strip any prose that precedes the first SQL keyword.
  5. If the prompt ended with "SELECT " (prebuilt-prompt / 1-shot mode),
     re-prepend "SELECT" when the model's continuation doesn't start with one.
  6. Take the first SQL statement (split on ';').
  7. Collapse whitespace and strip surrounding garbage.
"""
from __future__ import annotations

import re


_FENCE_RE     = re.compile(r"```(?:sql|sqlite)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_FINAL_SQL_RE = re.compile(r"final\s*sql\s*:?\s*(.+)",          re.IGNORECASE | re.DOTALL)
_SQL_START_RE = re.compile(r"\b(select|with|insert|update|delete)\b", re.IGNORECASE)
# ChatML tokens emitted by Qwen when the server doesn't enforce stop tokens.
_CHATML_RE       = re.compile(r"<\|im_end\|>.*", re.DOTALL)
_CHATML_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


def _strip_chatml(text: str) -> str:
    text = _CHATML_RE.sub("", text)
    text = _CHATML_TOKEN_RE.sub("", text)
    return text.strip()


def _extract_from_fence(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text


def _extract_final_sql(text: str) -> str:
    """CoT models end with `Final SQL: <query>` — grab the tail."""
    m = _FINAL_SQL_RE.search(text)
    if not m:
        return text
    candidate = m.group(1).strip()
    # The fragment may itself be fenced — peel once more.
    fenced = _extract_from_fence(candidate)
    return fenced if fenced != candidate else candidate


def _trim_to_first_sql_keyword(text: str) -> str:
    """Drop any prose that precedes the first SQL keyword.

    Handles responses that open with "Here is the SQL:", "Sure!", or a
    step-by-step explanation before the actual statement.
    """
    m = _SQL_START_RE.search(text)
    if m and m.start() > 0:
        return text[m.start():]
    return text


def _first_statement(text: str) -> str:
    """Return the first SQL statement with a single trailing semicolon."""
    text = text.strip().rstrip(";")
    head = text.split(";")[0].strip()
    return head + ";" if head else ""


def _collapse_whitespace(text: str) -> str:
    """Collapse whitespace runs outside single-quoted string literals."""
    out: list[str] = []
    in_str = False
    prev_space = False
    for ch in text:
        if ch == "'":
            in_str = not in_str
            out.append(ch)
            prev_space = False
        elif not in_str and ch in " \t\n\r":
            if not prev_space:
                out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out).strip()


def clean_sql(
    raw: str,
    *,
    cot: bool = False,
    completion_prefix: str = "",
) -> str:
    """Turn raw model output into a single executable SQL string."""
    if raw is None:
        return ""

    text = raw.strip()
    if not text:
        return ""

    # 1. Strip ChatML tokens.
    text = _strip_chatml(text)
    if not text:
        return ""

    # 2. Fenced code blocks win if present.
    fenced = _extract_from_fence(text)
    if fenced != text:
        text = fenced
    # 3. CoT sentinel extraction.
    elif cot:
        text = _extract_final_sql(text)

    # 4. Drop prose before the first SQL keyword (e.g. "Here is the query:").
    #    This also handles completion-prefix mode: when the model returns
    #    reasoning prose that embeds the real SELECT near the end, trimming
    #    moves the cursor to that SELECT so step 5 never needs to prepend.
    text = _trim_to_first_sql_keyword(text)

    # 5. Completion-prefix mode: prepend "SELECT" only when the model truly
    #    returned just the tail (e.g. "COUNT(*) FROM ...").
    if completion_prefix:
        if not re.match(r"^\s*select\b", text, re.IGNORECASE):
            text = completion_prefix + text

    # 6. Take the first statement.
    text = _first_statement(text)

    # 7. Collapse whitespace.
    text = _collapse_whitespace(text)
    return text
