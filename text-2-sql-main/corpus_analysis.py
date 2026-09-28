#!/usr/bin/env python3
"""
corpus_analysis.py
==================
Reproducible descriptive corpus-analysis pipeline for the Text2SQL4PM dataset.

Scope (per specification):
  - Dataset validation
  - Lexical diversity (TTR, MTLD)
  - Surface paraphrase similarity (Jaccard, anchor=_0)
  - SQL-grounded categorization (value, column, operation lexicalization)

Out of scope:
  - LLM inference, EX/EM evaluation, statistical significance, causal analysis.

Author: auto-generated 2026-08
"""
from __future__ import annotations

import csv
import json
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

# ---------------------------------------------------------------------------
# 0. CONFIGURATION  — edit paths here only
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DEV_JSON   = BASE_DIR / "data" / "dataset" / "english" / "dev.json"
GOLD_TXT   = BASE_DIR / "data" / "dataset" / "english" / "gold.txt"
TSV_PATH   = BASE_DIR / "data" / "dataset" / "text2sql4pm.tsv"
DB_PATH    = BASE_DIR / "data" / "dataset" / "english" / "event_log.sqlite"

OUTPUT_DIR = BASE_DIR / "corpus_analysis_results"

# Expected corpus constants 
EXPECTED_UTTERANCES   = 1655
EXPECTED_GROUPS       = 205

# MTLD threshold (canonical)
MTLD_THRESHOLD = 0.72

# ---------------------------------------------------------------------------
# 1. TEXT NORMALIZATION
# ---------------------------------------------------------------------------
# Rules (documented exhaustively):
#   - Lowercase: yes — reduces type count without semantic loss.
#   - Punctuation: removed (all characters in string.punctuation) — following
#     Spider / Text2SQL4PM question_toks convention.
#   - Contractions: not expanded — the dataset tokenization already splits
#     e.g. "don't" -> ["don", "'t"]; we strip the apostrophe and treat
#     each fragment as its own token. No external library.
#   - Numbers: preserved as-is — "2225", "3", "2018" are meaningful tokens.
#   - Schema identifiers: preserved — "idcase", "resource", "activity" etc.
#     are content words in this domain; stripping them would distort TTR.
#   - Stemming / lemmatization: NONE — not required by the existing
#     dataset pre-processing.
# ---------------------------------------------------------------------------
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace. Return token list."""
    lowered = text.lower()
    stripped = lowered.translate(_PUNCT_TABLE)
    return [t for t in stripped.split() if t]


def token_set(text: str) -> set[str]:
    """Return the set of unique normalized tokens in *text*."""
    return set(normalize_text(text))


# ---------------------------------------------------------------------------
# 2. MTLD IMPLEMENTATION
# ---------------------------------------------------------------------------
# Forward pass:
#   Scan tokens left-to-right keeping a running set of unique types.
#   When running_TTR <= threshold:
#     increment factor count, reset the running set.
#   At the end, add a partial factor = (1 - TTR_residual) / (1 - threshold).
# Backward pass: repeat on reversed token list.
# Final MTLD = total_tokens / mean(forward_factors, backward_factors).
# ---------------------------------------------------------------------------

def _mtld_single_pass(tokens: list[str], threshold: float) -> float:
    """Compute single-direction MTLD factor count (may be fractional)."""
    if not tokens:
        return 0.0
    factors = 0
    types: set[str] = set()
    token_count = 0
    for tok in tokens:
        types.add(tok)
        token_count += 1
        ttr = len(types) / token_count
        if ttr <= threshold:
            factors += 1
            types = set()
            token_count = 0
    # Partial factor for residual segment
    if token_count > 0:
        residual_ttr = len(types) / token_count
        if residual_ttr < 1.0:  # avoid division by zero when residual is a single new type
            partial = (1.0 - residual_ttr) / (1.0 - threshold)
        else:
            partial = 0.0
        factors += partial
    return factors


def compute_mtld(tokens: list[str], threshold: float = MTLD_THRESHOLD) -> float:
    """
    Compute MTLD (Measure of Textual Lexical Diversity).

    Uses bidirectional computation (McCarthy & Jarvis 2010):
      MTLD = len(tokens) / mean(forward_factors, backward_factors)

    Returns 0.0 for corpora shorter than 2 tokens.
    """
    if len(tokens) < 2:
        return 0.0
    fwd = _mtld_single_pass(tokens, threshold)
    bwd = _mtld_single_pass(list(reversed(tokens)), threshold)
    avg_factors = (fwd + bwd) / 2.0
    if avg_factors == 0:
        return 0.0
    return len(tokens) / avg_factors


# ---------------------------------------------------------------------------
# 3. SQL PARSING UTILITIES
# ---------------------------------------------------------------------------
# We use regex-based extraction. Limitations are documented at each use site.
# A proper SQL parser (e.g. sqlglot) was not present in requirements.txt;
# the dataset uses a small, predictable SQL fragment so regex is sufficient
# here, but we log all failures.
# ---------------------------------------------------------------------------

# Schema columns (authoritative)
SCHEMA_COLUMNS = {"id", "activity", "timestamp", "resource", "cost", "idcase"}

# Analysed columns (id is a surrogate key, not semantically queried)
TARGET_COLUMNS = ["idcase", "activity", "resource", "timestamp", "cost"]


def sql_upper(sql: str) -> str:
    """Return SQL with keywords uppercased (preserves string literals)."""
    return sql  # We use case-insensitive regex below


def extract_columns_referenced(sql: str) -> set[str]:
    """
    Return the set of target schema columns that appear in the SQL.
    A column is considered referenced if it appears outside of quoted literals.
    Limitation: uses simple regex; may misfire on column names that also appear
    inside string literals (rare in this dataset).
    """
    # Remove string literals to avoid false positives
    sql_no_strings = re.sub(r"'[^']*'", "''", sql)
    sql_no_strings = re.sub(r'"[^"]*"', '""', sql_no_strings)
    referenced = set()
    for col in TARGET_COLUMNS:
        # Word-boundary match, case-insensitive
        if re.search(rf"\b{re.escape(col)}\b", sql_no_strings, re.IGNORECASE):
            referenced.add(col)
    return referenced


def extract_literal_values(sql: str) -> list[dict]:
    """
    Extract literal values from a SQL query.
    Returns list of dicts: {literal, value_type}

    Detected types:
      - string       : single-quoted or double-quoted string literals
      - numeric      : bare numbers (integer or decimal)
      - date_string  : single-quoted values matching date/datetime patterns

    Limitation: does not parse nested subqueries specially. The patterns
    cover the full range of literal forms observed in this dataset.
    """
    literals = []
    # ---- String / date literals (single-quoted)
    for m in re.finditer(r"'([^']+)'", sql):
        val = m.group(1)
        # Date or datetime pattern
        if re.match(r"\d{4}-\d{2}-\d{2}", val):
            literals.append({"literal": val, "value_type": "date_string"})
        # Year-month pattern used in strftime comparisons
        elif re.match(r"\d{4}-\d{2}$", val):
            literals.append({"literal": val, "value_type": "date_string"})
        # Year-pattern LIKE e.g. '2016%' 
        elif re.match(r"\d{4}%", val):
            literals.append({"literal": val.rstrip('%'), "value_type": "year_pattern"})
        else:
            # Domain string (activity name, resource name, case ID, etc.)
            literals.append({"literal": val, "value_type": "string"})

    # ---- Double-quoted string literals (some queries use double quotes)
    for m in re.finditer(r'"([^"]+)"', sql):
        val = m.group(1)
        if re.match(r"\d{4}-\d{2}-\d{2}", val):
            literals.append({"literal": val, "value_type": "date_string"})
        else:
            literals.append({"literal": val, "value_type": "string"})

    # ---- Bare numeric literals (not inside quotes, not column names)
    sql_no_strings = re.sub(r"'[^']*'", "''", sql)
    sql_no_strings = re.sub(r'"[^"]*"', '""', sql_no_strings)
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", sql_no_strings):
        num_str = m.group(1)
        # Exclude "1" when it's the literal in COUNT(*) = (... LIMIT 1)
        # or percentages used in LIMIT; we include ALL numbers as per methodology
        literals.append({"literal": num_str, "value_type": "numeric"})

    return literals


def detect_sql_operations(sql: str) -> dict[str, bool]:
    """
    Detect which SQL operations are present.
    Returns dict: {operation_name: bool}.

    Operations detected (case-insensitive, whole-word):
      COUNT, DISTINCT, SUM, MIN, MAX, GROUP BY, Temporal, Negation, Set operation
    """
    sql_up = sql.upper()
    sql_no_str = re.sub(r"'[^']*'", "''", sql_up)
    sql_no_str = re.sub(r'"[^"]*"', '""', sql_no_str)

    ops: dict[str, bool] = {}

    ops["COUNT"] = bool(re.search(r"\bCOUNT\s*\(", sql_no_str))
    # DISTINCT: both COUNT(DISTINCT...) and SELECT DISTINCT
    ops["DISTINCT"] = bool(re.search(r"\bDISTINCT\b", sql_no_str))
    ops["SUM"] = bool(re.search(r"\bSUM\s*\(", sql_no_str))
    ops["MIN"] = bool(re.search(r"\bMIN\s*\(", sql_no_str))
    ops["MAX"] = bool(re.search(r"\bMAX\s*\(", sql_no_str))
    ops["GROUP BY"] = bool(re.search(r"\bGROUP\s+BY\b", sql_no_str))

    # Temporal: BETWEEN...AND, timestamp LIKE, strftime, or explicit date comparisons
    has_temporal = (
        bool(re.search(r"\bBETWEEN\b", sql_no_str))
        or bool(re.search(r"\bstrftime\b", sql_no_str))
        or bool(re.search(r"\btimestamp\b.*\bLIKE\b", sql_no_str, re.IGNORECASE))
        or bool(re.search(r"\btimestamp\b.*[<>=]", sql_no_str))
    )
    ops["Temporal"] = has_temporal

    # Negation: != , <>, NOT IN, NOT NULL, IS NOT NULL
    has_neg = (
        bool(re.search(r"!=|<>", sql_no_str))
        or bool(re.search(r"\bNOT\b", sql_no_str))
    )
    ops["Negation"] = has_neg

    # Set operations: EXCEPT, INTERSECT, UNION
    ops["Set operation"] = bool(re.search(r"\b(EXCEPT|INTERSECT|UNION)\b", sql_no_str))

    return ops


# ---------------------------------------------------------------------------
# 4. COLUMN SYNONYM DICTIONARY
# ---------------------------------------------------------------------------
# Derived from the project's LEXICON BLOCK (_LEXICON_BLOCK in experiment.py).
# Trigger words from that block that map to schema columns:
#
#  Case -> idcase (the Case trigger words refer to the case/process-instance,
#         which is stored in idcase)
#  Activity -> activity column
#  Resource -> resource column  
#  timestamp -> temporal references
#  cost -> cost references
#
# Only the trigger words explicitly listed in the project lexicon are used.
# NO additional synonyms from WordNet, embeddings, or LLM are added.
# ---------------------------------------------------------------------------

COLUMN_SYNONYMS: dict[str, list[str]] = {
    "idcase": [
        # Literal column name
        "idcase",
        # From _LEXICON_BLOCK Case trigger words + experiment conventions
        "case", "process instance", "declaration", "travel declaration",
        "permit", "trip", "process", "identifier", "id",
    ],
    "activity": [
        # Literal column name
        "activity", "activities",
        # From _LEXICON_BLOCK Activity trigger words
        "task", "action",
        # Partial matches (when used as nouns referring to the column)
        "event",  # included here because events map to activity rows
    ],
    "resource": [
        # Literal column name
        "resource", "resources",
        # From _LEXICON_BLOCK Resource trigger words
        "employee", "employees", "person", "people", "collaborator",
        "supervisor", "director", "administration", "administrator",
        "pre-approver", "approver", "budget owner", "budget holder",
        "worker", "staff",
    ],
    "timestamp": [
        # Literal column name
        "timestamp",
        # Derived from Time Predicates in _LOGICAL_FORM_BLOCK
        "date", "time", "day", "month", "year", "when", "period",
        "temporal", "duration", "period",
    ],
    "cost": [
        # Literal column name
        "cost", "costs",
        # From Cost Predicates in _LOGICAL_FORM_BLOCK
        "expense", "expenses", "price", "amount", "fee", "payment",
    ],
}

# Pre-normalize synonym lists
_COLUMN_SYNONYM_SETS: dict[str, set[str]] = {
    col: {s.lower() for s in syns}
    for col, syns in COLUMN_SYNONYMS.items()
}


def utterance_lexicalizes_column(utterance: str, column: str) -> bool:
    """
    Return True if *utterance* contains the column name or any synonym.
    Matching is done against the normalized utterance at the word-token level
    (so 'costs' matches 'cost', but 'process' does not match 'pro').
    Multi-word synonyms are matched as substrings of the lowercased utterance
    (acceptable because they are domain-specific phrases unlikely to create
    false positives).
    """
    lower_utt = utterance.lower()
    tokens = set(normalize_text(utterance))
    syns = _COLUMN_SYNONYM_SETS[column]
    for s in syns:
        if " " in s:
            # Multi-word: substring match on lowercased utterance
            if s in lower_utt:
                return True
        else:
            if s in tokens:
                return True
    return False


# ---------------------------------------------------------------------------
# 5. OPERATION NL-CUE DICTIONARY
# ---------------------------------------------------------------------------
# Derived from the _LOGICAL_FORM_BLOCK cues in experiment.py.
# Only patterns explicitly listed in the project are used.
# Cues are regex patterns applied to the lowercased utterance.
# ---------------------------------------------------------------------------

OPERATION_CUES: dict[str, list[str]] = {
    "COUNT": [
        r"\bhow many\b",
        r"\bnumber of\b",
        r"\bcount\b",
        r"\bquantify\b",
        r"\bhow many times\b",
        r"\boccurrences\b",
        r"\boccurrence\b",
    ],
    "DISTINCT": [
        r"\bdifferent\b",
        r"\bdistinct\b",
        r"\bunique\b",
        r"\bdifferentes\b",
    ],
    "SUM": [
        r"\btotal\b",
        r"\bsum\b",
        r"\baggregat\b",
    ],
    "MIN": [
        r"\bminimum\b",
        r"\bmin\b",
        r"\bsmallest\b",
        r"\blowest\b",
        r"\bfewest\b",
        r"\bleast\b",
    ],
    "MAX": [
        r"\bmaximum\b",
        r"\bmax\b",
        r"\blargest\b",
        r"\bhighest\b",
        r"\bmost\b",
        r"\bgreatest\b",
    ],
    "GROUP BY": [
        r"\beach\b",
        r"\bper\b",
        r"\bby\b",
        r"\bgroup\b",
        r"\bbreakdown\b",
        r"\bbreaking down\b",
        r"\bdistribution\b",
    ],
    "Temporal": [
        r"\bdate\b",
        r"\btime\b",
        r"\bwhen\b",
        r"\bday\b",
        r"\bmonth\b",
        r"\byear\b",
        r"\bbetween\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\bduring\b",
        r"\bperiod\b",
        r"\bin \d{4}\b",
        r"\b20\d{2}\b",
        r"\bjanuary|february|march|april|may|june|july|august|september|october|november|december\b",
    ],
    "Negation": [
        r"\bnot\b",
        r"\bexcept\b",
        r"\bexcluding\b",
        r"\bother than\b",
        r"\bwithout\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bnone\b",
        r"\bdifferent from\b",
    ],
    "Set operation": [
        r"\bexcept\b",
        r"\bunion\b",
        r"\bintersect\b",
        r"\bbut not\b",
        r"\bnot in\b",
    ],
}


def utterance_has_nl_cue(utterance: str, operation: str) -> bool:
    """Return True if *utterance* contains any NL cue for *operation*."""
    lower_utt = utterance.lower()
    for pattern in OPERATION_CUES.get(operation, []):
        if re.search(pattern, lower_utt):
            return True
    return False


# ---------------------------------------------------------------------------
# 6. VALUE LEXICALIZATION HELPERS
# ---------------------------------------------------------------------------

def _normalize_value_for_match(val: str) -> str:
    """Lowercase and strip outer whitespace for comparison."""
    return val.strip().lower()


def _check_string_value(val: str, utterance: str) -> bool:
    """
    Check if string literal *val* appears in *utterance*.
    Case-insensitive. Both exact and substring match are accepted
    (a string value may be quoted differently in the utterance).
    """
    val_norm = _normalize_value_for_match(val)
    utt_norm = utterance.lower()
    # Direct substring check
    if val_norm in utt_norm:
        return True
    # Also try without apostrophes/hyphens
    val_clean = re.sub(r"['-]", " ", val_norm).strip()
    utt_clean = re.sub(r"['-]", " ", utt_norm).strip()
    return val_clean in utt_clean


def _check_numeric_value(val: str, utterance: str) -> bool:
    """
    Check if numeric literal *val* appears in *utterance*.
    Accepts both digit and common textual representations.
    '1' is excluded as it is a SQL artifact (LIMIT 1, = 1 in HAVING).
    """
    if val in {"1"}:  # Ubiquitous SQL artifact; skip
        return False
    # Check digit form
    utt_lower = utterance.lower()
    # Word-boundary match to avoid '10' matching inside '100'
    if re.search(rf"\b{re.escape(val)}\b", utt_lower):
        return True
    # Check for European decimal notation e.g. 2.225 -> 2225 in query
    if "." not in val and len(val) >= 4:
        # Try formatted with comma or period thousands separator
        formatted_comma = f"{int(val):,}"  # e.g. "2,225"
        formatted_period = formatted_comma.replace(",", ".")  # e.g. "2.225"
        if formatted_comma in utt_lower or formatted_period in utt_lower:
            return True
    return False


def _check_date_value(val: str, utterance: str) -> bool:
    """
    Check if date literal *val* appears in utterance.
    Accepts:
      - exact string match
      - year alone
      - year-month in various formats (March 2018, 2018-03, etc.)
    """
    utt_lower = utterance.lower()
    val_norm = val.strip().lower()
    if val_norm in utt_lower:
        return True

    # Try individual date components
    parts = val_norm.split("-")
    year = parts[0] if parts else ""
    month_num = parts[1] if len(parts) > 1 else ""
    day_num = parts[2] if len(parts) > 2 else ""

    if year and year in utt_lower:
        return True

    # Month name match
    _months = {
        "01": "january", "02": "february", "03": "march",
        "04": "april", "05": "may", "06": "june",
        "07": "july", "08": "august", "09": "september",
        "10": "october", "11": "november", "12": "december",
    }
    if month_num in _months:
        if _months[month_num] in utt_lower:
            return True
        if month_num.lstrip("0") in utt_lower:
            return True

    if day_num and day_num.lstrip("0") in utt_lower:
        return True

    return False


def check_value_lexicalized(literal_info: dict, utterance: str) -> bool:
    """
    Route to the appropriate matching function based on value_type.
    Numeric literal '1' is always excluded (SQL artifact).
    """
    val = literal_info["literal"]
    vtype = literal_info["value_type"]
    if vtype == "string":
        return _check_string_value(val, utterance)
    elif vtype in ("date_string", "year_pattern"):
        return _check_date_value(val, utterance)
    elif vtype == "numeric":
        return _check_numeric_value(val, utterance)
    return False


# ---------------------------------------------------------------------------
# 7. DATA LOADING
# ---------------------------------------------------------------------------

def load_dev_json(path: Path) -> list[dict]:
    """Load dev.json. Returns list of utterance dicts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] dev.json: {len(data)} records loaded.")
    return data


def load_gold_txt(path: Path) -> dict[str, str]:
    """
    Load gold.txt.
    Format per line: <sql>\t<db_id>\t<utterance_id>
    Returns dict {utterance_id: sql}.
    """
    result = {}
    parse_failures = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parse_failures.append(f"gold.txt line {lineno}: expected 3 tab-fields, got {len(parts)}")
                continue
            sql, db_id, sid = parts[0], parts[1], parts[2]
            result[sid] = sql
    if parse_failures:
        print(f"[WARN] gold.txt parse failures ({len(parse_failures)}):")
        for f in parse_failures:
            print(f"  {f}")
    print(f"[INFO] gold.txt: {len(result)} entries loaded.")
    return result


def load_tsv(path: Path) -> dict[str, dict]:
    """
    Load text2sql4pm.tsv.
    Returns dict keyed by constructed utterance_id = f"{group_id}_{position}".
    Empty-utterance rows (Portuguese-only) are skipped.
    Within each group, rows are sorted by Utterance_id (integer) to match
    dev.json positional numbering.
    """
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        eng = r.get("English_utterance", "").strip()
        if not eng:
            continue
        gid = str(r.get("Group_id", "")).strip()
        if not gid:
            continue
        grouped[gid].append(r)

    for gid in grouped:
        grouped[gid].sort(key=lambda r: int(r["Utterance_id"]))

    out: dict[str, dict] = {}
    for gid, items in grouped.items():
        for pos, row in enumerate(items):
            sid = f"{gid}_{pos}"
            out[sid] = row

    print(f"[INFO] text2sql4pm.tsv: {len(out)} English-utterance rows loaded.")
    return out


def build_corpus(dev: list[dict], gold: dict[str, str], tsv: dict[str, dict]) -> list[dict]:
    """
    Merge dev.json, gold.txt, and tsv into a unified corpus record per utterance.
    Returns list of dicts with keys:
      utterance_id, group_id, question, question_toks, gold_sql, u_class,
      spider_class, domain_value_class
    Unmatched IDs are logged.
    """
    corpus = []
    unmatched_gold = []
    unmatched_tsv = []

    for item in dev:
        sid = item["id"]
        gid = str(item.get("group_id", sid.split("_")[0]))
        question = item["question"]
        question_toks = item.get("question_toks", [])

        # Gold SQL: prefer gold.txt (external reference), fall back to dev.json
        gold_sql = gold.get(sid)
        if gold_sql is None:
            gold_sql = item.get("query", "")
            unmatched_gold.append(sid)

        # TSV metadata
        tsv_row = tsv.get(sid)
        if tsv_row is None:
            unmatched_tsv.append(sid)
            spider_class = "no_hardness"
            domain_value_class = ""
        else:
            spider_class = tsv_row.get("Spider_classification", "no_hardness")
            domain_value_class = tsv_row.get("Domain_value_generic", "")

        corpus.append({
            "utterance_id": sid,
            "group_id": gid,
            "question": question,
            "question_toks": question_toks,
            "gold_sql": gold_sql,
            "u_class": item.get("u_class", ""),
            "spider_class": spider_class,
            "domain_value_class": domain_value_class,
        })

    if unmatched_gold:
        print(f"[WARN] {len(unmatched_gold)} utterance IDs not found in gold.txt "
              f"(using dev.json query field as fallback).")
        if len(unmatched_gold) <= 20:
            for s in unmatched_gold:
                print(f"  missing in gold.txt: {s}")

    if unmatched_tsv:
        print(f"[WARN] {len(unmatched_tsv)} utterance IDs not found in text2sql4pm.tsv "
              f"(hardness/class set to defaults).")
        if len(unmatched_tsv) <= 20:
            for s in unmatched_tsv:
                print(f"  missing in tsv: {s}")

    return corpus


# ---------------------------------------------------------------------------
# 8. HELPERS
# ---------------------------------------------------------------------------

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def pct(a: float, b: float) -> float:
    return round(safe_div(a, b) * 100, 2)


def describe(values: list[float]) -> dict:
    """Return descriptive stats dict for a list of floats."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "sd": None, "min": None, "max": None}
    n = len(values)
    mu = mean(values)
    med = median(values)
    sd = stdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mu, 6),
        "median": round(med, 6),
        "sd": round(sd, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def describe_with_quartiles(values: list[float]) -> dict:
    """Return descriptive stats including Q1 and Q3."""
    d = describe(values)
    if not values:
        d["q1"] = None
        d["q3"] = None
        return d
    sorted_v = sorted(values)
    n = len(sorted_v)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[(3 * n) // 4]
    d["q1"] = round(q1, 6)
    d["q3"] = round(q3, 6)
    return d


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        print(f"[WARN] No rows to write to {path.name}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Written {len(rows)} rows -> {path.name}")


# ---------------------------------------------------------------------------
# 9. STAGE 1: DATASET VALIDATION
# ---------------------------------------------------------------------------

def stage1_validation(corpus: list[dict]) -> dict:
    """
    Validates:
      1. Corpus size = 1655 utterances
      2. Number of paraphrase groups = 205
      3. Group size statistics
      4. Gold SQL consistency within each group
    """
    print("\n" + "="*60)
    print("STAGE 1: DATASET VALIDATION")
    print("="*60)

    n_utterances = len(corpus)
    print(f"\nTotal utterances in corpus: {n_utterances}")
    if n_utterances != EXPECTED_UTTERANCES:
        print(f"[DISCREPANCY] Expected {EXPECTED_UTTERANCES}, got {n_utterances}!")
    else:
        print(f"[OK] Utterance count matches expected {EXPECTED_UTTERANCES}.")

    # Group by group_id
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in corpus:
        groups[item["group_id"]].append(item)

    n_groups = len(groups)
    print(f"Total paraphrase groups: {n_groups}")
    if n_groups != EXPECTED_GROUPS:
        print(f"[DISCREPANCY] Expected {EXPECTED_GROUPS}, got {n_groups}!")
    else:
        print(f"[OK] Group count matches expected {EXPECTED_GROUPS}.")

    # Group size statistics
    group_sizes = [len(v) for v in groups.values()]
    gs_mean   = mean(group_sizes)
    gs_median = median(group_sizes)
    gs_min    = min(group_sizes)
    gs_max    = max(group_sizes)
    print(f"\nGroup-size statistics:")
    print(f"  Mean   : {gs_mean:.2f}")
    print(f"  Median : {gs_median}")
    print(f"  Min    : {gs_min}")
    print(f"  Max    : {gs_max}")

    # Gold SQL consistency
    def normalize_sql(sql: str) -> str:
        """Normalize SQL for consistency comparison: strip whitespace, lowercase."""
        return re.sub(r"\s+", " ", sql.strip().lower())

    inconsistent_groups = []
    for gid, items in groups.items():
        gold_sqls = {normalize_sql(item["gold_sql"]) for item in items}
        if len(gold_sqls) > 1:
            inconsistent_groups.append({
                "group_id": gid,
                "n_distinct_gold_sqls": len(gold_sqls),
                "distinct_sqls": " ||| ".join(sorted(gold_sqls)),
            })

    n_inconsistent = len(inconsistent_groups)
    print(f"\nGroups with >1 distinct normalized gold SQL: {n_inconsistent}")
    if n_inconsistent > 0:
        print(f"[REPORT] The following groups have inconsistent gold SQL queries:")
        for ig in inconsistent_groups:
            print(f"  group {ig['group_id']}: {ig['n_distinct_gold_sqls']} distinct queries")

    # Build validation rows for CSV
    validation_rows = []
    for gid, items in sorted(groups.items(), key=lambda x: int(x[0])):
        gold_sqls_raw = [item["gold_sql"] for item in items]
        gold_sqls_norm = {normalize_sql(s) for s in gold_sqls_raw}
        validation_rows.append({
            "group_id": gid,
            "n_utterances": len(items),
            "n_distinct_gold_sql": len(gold_sqls_norm),
            "gold_sql_consistent": len(gold_sqls_norm) == 1,
            "gold_sql_sample": gold_sqls_raw[0][:200],
        })

    summary = {
        "n_utterances": n_utterances,
        "n_groups": n_groups,
        "utterance_count_ok": n_utterances == EXPECTED_UTTERANCES,
        "group_count_ok": n_groups == EXPECTED_GROUPS,
        "group_size_mean": round(gs_mean, 4),
        "group_size_median": gs_median,
        "group_size_min": gs_min,
        "group_size_max": gs_max,
        "n_inconsistent_groups": n_inconsistent,
        "groups": groups,
        "validation_rows": validation_rows,
        "inconsistent_groups": inconsistent_groups,
    }
    return summary


# ---------------------------------------------------------------------------
# 10. STAGE 2: LEXICAL DIVERSITY
# ---------------------------------------------------------------------------

def stage2_lexical_diversity(corpus: list[dict], groups: dict[str, list[dict]]) -> dict:
    """
    Computes corpus-level and group-level TTR and MTLD.
    Tokenization: question_toks from dev.json (already tokenized by the dataset),
    then normalized (lowercase, punct-stripped).
    """
    print("\n" + "="*60)
    print("STAGE 2: LEXICAL DIVERSITY")
    print("="*60)

    # --- 2.1 Corpus-level ---
    # Concatenate all utterances (using question_toks from dev.json,
    # then applying our normalization)
    all_tokens: list[str] = []
    for item in corpus:
        toks = item.get("question_toks", [])
        if toks:
            # question_toks already tokenized; apply normalization
            all_tokens.extend(normalize_text(" ".join(toks)))
        else:
            all_tokens.extend(normalize_text(item["question"]))

    n_tokens = len(all_tokens)
    n_types  = len(set(all_tokens))
    corpus_ttr  = safe_div(n_types, n_tokens)
    corpus_mtld = compute_mtld(all_tokens, MTLD_THRESHOLD)

    print(f"\nCorpus-level lexical diversity (N={n_tokens} tokens):")
    print(f"  Total tokens      : {n_tokens}")
    print(f"  Unique word types : {n_types}")
    print(f"  Corpus TTR        : {corpus_ttr:.6f}")
    print(f"  Corpus MTLD       : {corpus_mtld:.4f}  (threshold={MTLD_THRESHOLD})")

    print(f"\nText normalization rules applied:")
    print(f"  - Lowercase: yes")
    print(f"  - Punctuation removal: yes (string.punctuation stripped)")
    print(f"  - Contractions: not expanded (apostrophes removed, fragments kept)")
    print(f"  - Numbers: preserved as-is")
    print(f"  - Schema identifiers: preserved")
    print(f"  - Stemming/lemmatization: none")
    print(f"  - Token source: question_toks field from dev.json (pre-tokenized)")

    corpus_summary = [{
        "level": "corpus",
        "n_tokens": n_tokens,
        "n_types": n_types,
        "ttr": round(corpus_ttr, 6),
        "mtld": round(corpus_mtld, 4),
        "mtld_threshold": MTLD_THRESHOLD,
    }]

    # --- 2.2 Group-level ---
    group_lex_rows = []
    group_ttrs  = []
    group_mtlds = []

    for gid, items in sorted(groups.items(), key=lambda x: int(x[0])):
        # Concatenate all utterances in the group
        group_toks: list[str] = []
        for item in items:
            toks = item.get("question_toks", [])
            if toks:
                group_toks.extend(normalize_text(" ".join(toks)))
            else:
                group_toks.extend(normalize_text(item["question"]))

        g_n_tokens = len(group_toks)
        g_n_types  = len(set(group_toks))
        g_ttr      = safe_div(g_n_types, g_n_tokens)
        g_mtld     = compute_mtld(group_toks, MTLD_THRESHOLD)

        group_ttrs.append(g_ttr)
        group_mtlds.append(g_mtld)

        group_lex_rows.append({
            "group_id": gid,
            "n_utterances": len(items),
            "n_tokens": g_n_tokens,
            "n_types": g_n_types,
            "ttr": round(g_ttr, 6),
            "mtld": round(g_mtld, 4),
        })

    ttr_stats  = describe(group_ttrs)
    mtld_stats = describe(group_mtlds)

    print(f"\nGroup-level TTR  (N={len(group_ttrs)} groups):")
    print(f"  Mean={ttr_stats['mean']:.4f}  SD={ttr_stats['sd']:.4f}  "
          f"Median={ttr_stats['median']:.4f}  "
          f"Range=[{ttr_stats['min']:.4f}, {ttr_stats['max']:.4f}]")

    print(f"Group-level MTLD (N={len(group_mtlds)} groups):")
    print(f"  Mean={mtld_stats['mean']:.4f}  SD={mtld_stats['sd']:.4f}  "
          f"Median={mtld_stats['median']:.4f}  "
          f"Range=[{mtld_stats['min']:.4f}, {mtld_stats['max']:.4f}]")

    return {
        "corpus_summary": corpus_summary,
        "corpus_ttr": corpus_ttr,
        "corpus_mtld": corpus_mtld,
        "n_tokens": n_tokens,
        "n_types": n_types,
        "group_lex_rows": group_lex_rows,
        "ttr_stats": ttr_stats,
        "mtld_stats": mtld_stats,
    }


# ---------------------------------------------------------------------------
# 11. STAGE 3: SURFACE PARAPHRASE SIMILARITY
# ---------------------------------------------------------------------------

def stage3_jaccard(groups: dict[str, list[dict]]) -> dict:
    """
    Compute Jaccard token similarity between each group's _0 anchor and
    every other variant in the group.
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    print("\n" + "="*60)
    print("STAGE 3: SURFACE PARAPHRASE SIMILARITY (Jaccard)")
    print("="*60)

    pair_rows: list[dict] = []
    group_mean_rows: list[dict] = []
    skipped_groups_no_anchor = []

    for gid, items in sorted(groups.items(), key=lambda x: int(x[0])):
        # Find anchor (_0)
        anchor = None
        for item in items:
            if item["utterance_id"] == f"{gid}_0":
                anchor = item
                break
        if anchor is None:
            skipped_groups_no_anchor.append(gid)
            continue

        anchor_tokens = token_set(anchor["question"])
        group_jaccard_values = []

        for item in items:
            if item["utterance_id"] == f"{gid}_0":
                continue  # Skip anchor vs itself
            var_tokens = token_set(item["question"])
            union = anchor_tokens | var_tokens
            inter = anchor_tokens & var_tokens
            j = safe_div(len(inter), len(union))
            group_jaccard_values.append(j)
            pair_rows.append({
                "group_id": gid,
                "anchor_id": anchor["utterance_id"],
                "variant_id": item["utterance_id"],
                "anchor_text": anchor["question"],
                "variant_text": item["question"],
                "jaccard_similarity": round(j, 6),
            })

        group_mean_j = mean(group_jaccard_values) if group_jaccard_values else None
        group_mean_rows.append({
            "group_id": gid,
            "n_comparisons": len(group_jaccard_values),
            "mean_jaccard": round(group_mean_j, 6) if group_mean_j is not None else None,
        })

    if skipped_groups_no_anchor:
        print(f"[WARN] {len(skipped_groups_no_anchor)} groups had no _0 anchor: "
              f"{skipped_groups_no_anchor}")

    all_j = [r["jaccard_similarity"] for r in pair_rows]
    jac_stats = describe_with_quartiles(all_j)

    print(f"\nAnchor-variant comparisons: {len(pair_rows)}")
    print(f"  Mean Jaccard   : {jac_stats['mean']:.4f}")
    print(f"  SD             : {jac_stats['sd']:.4f}")
    print(f"  Median         : {jac_stats['median']:.4f}")
    print(f"  Q1             : {jac_stats['q1']:.4f}")
    print(f"  Q3             : {jac_stats['q3']:.4f}")
    print(f"  Range          : [{jac_stats['min']:.4f}, {jac_stats['max']:.4f}]")

    return {
        "pair_rows": pair_rows,
        "group_mean_rows": group_mean_rows,
        "jac_stats": jac_stats,
    }


# ---------------------------------------------------------------------------
# 12. STAGE 4: SQL-GROUNDED CATEGORIZATION
# ---------------------------------------------------------------------------

def stage4_value_lexicalization(corpus: list[dict]) -> dict:
    """
    For each utterance, extract all literal values from gold SQL and check
    whether they appear in the utterance.
    """
    print("\n" + "="*60)
    print("STAGE 4.1: VALUE LEXICALIZATION")
    print("="*60)

    value_rows: list[dict] = []
    utterance_val_summary: dict[str, dict] = {}

    sql_parse_failures = []

    for item in corpus:
        sid = item["utterance_id"]
        gid = item["group_id"]
        sql = item["gold_sql"]
        question = item["question"]

        try:
            literals = extract_literal_values(sql)
        except Exception as e:
            sql_parse_failures.append(f"{sid}: {e}")
            literals = []

        # Filter: exclude '1' numeric artifact at record level
        # (it is already excluded inside _check_numeric_value for individual checks)
        applicable_literals = [
            l for l in literals
            if not (l["value_type"] == "numeric" and l["literal"] == "1")
        ]

        n_applicable = len(applicable_literals)
        n_lex = 0
        for lit in applicable_literals:
            lex = check_value_lexicalized(lit, question)
            if lex:
                n_lex += 1
            value_rows.append({
                "utterance_id": sid,
                "group_id": gid,
                "gold_sql": sql,
                "literal_value": lit["literal"],
                "value_type": lit["value_type"],
                "value_lexicalized": lex,
            })

        utterance_val_summary[sid] = {
            "utterance_id": sid,
            "group_id": gid,
            "has_literals": n_applicable > 0,
            "n_literals": n_applicable,
            "n_lexicalized": n_lex,
            "n_not_lexicalized": n_applicable - n_lex,
        }

    if sql_parse_failures:
        print(f"[WARN] {len(sql_parse_failures)} SQL literal-extraction failures:")
        for f in sql_parse_failures[:10]:
            print(f"  {f}")

    # Aggregate
    applicable_utts = [v for v in utterance_val_summary.values() if v["has_literals"]]
    n_applicable_utts = len(applicable_utts)
    total_literals = sum(r["n_literals"] for r in applicable_utts)
    total_lex      = sum(r["n_lexicalized"] for r in applicable_utts)
    total_not_lex  = total_literals - total_lex

    print(f"\nApplicable utterances (have >=1 literal): {n_applicable_utts}")
    print(f"Total literal instances: {total_literals}")
    print(f"Lexicalized: {total_lex} ({pct(total_lex, total_literals):.1f}%)")
    print(f"Not lexicalized: {total_not_lex} ({pct(total_not_lex, total_literals):.1f}%)")

    summary_rows = [{
        "metric": "applicable_utterances",
        "value": n_applicable_utts,
    }, {
        "metric": "total_literal_instances",
        "value": total_literals,
    }, {
        "metric": "n_lexicalized",
        "value": total_lex,
    }, {
        "metric": "pct_lexicalized",
        "value": pct(total_lex, total_literals),
    }, {
        "metric": "n_not_lexicalized",
        "value": total_not_lex,
    }, {
        "metric": "pct_not_lexicalized",
        "value": pct(total_not_lex, total_literals),
    }]

    return {
        "value_rows": value_rows,
        "utterance_val_summary": utterance_val_summary,
        "summary_rows": summary_rows,
        "n_applicable_utts": n_applicable_utts,
        "total_literals": total_literals,
        "total_lex": total_lex,
        "total_not_lex": total_not_lex,
    }


def stage4_column_lexicalization(corpus: list[dict]) -> dict:
    """
    For each utterance, identify referenced schema columns and check
    whether each appears in the utterance via the synonym dictionary.
    """
    print("\n" + "="*60)
    print("STAGE 4.2: COLUMN LEXICALIZATION")
    print("="*60)

    col_rows: list[dict] = []

    for item in corpus:
        sid  = item["utterance_id"]
        gid  = item["group_id"]
        sql  = item["gold_sql"]
        q    = item["question"]
        referenced = extract_columns_referenced(sql)
        for col in TARGET_COLUMNS:
            if col not in referenced:
                status = "NA"  # column not referenced in SQL
                lex = None
            else:
                lex = utterance_lexicalizes_column(q, col)
                status = "lexicalized" if lex else "not_lexicalized"
            col_rows.append({
                "utterance_id": sid,
                "group_id": gid,
                "column": col,
                "sql_references_column": col in referenced,
                "column_lexicalized": lex,   # None = NA
                "status": status,
            })

    # Per-column aggregates
    col_summary_rows = []
    col_stats: dict[str, dict] = {}
    for col in TARGET_COLUMNS:
        col_data = [r for r in col_rows if r["column"] == col]
        referenced_data = [r for r in col_data if r["sql_references_column"]]
        n_ref = len(referenced_data)
        n_lex = sum(1 for r in referenced_data if r["column_lexicalized"] is True)
        n_not = n_ref - n_lex
        rate  = pct(n_lex, n_ref)
        col_stats[col] = {
            "column": col,
            "references": n_ref,
            "lexicalized": n_lex,
            "not_lexicalized": n_not,
            "lexicalization_rate_pct": rate,
        }
        col_summary_rows.append(col_stats[col])

    # Overall rate
    total_ref = sum(v["references"] for v in col_stats.values())
    total_lex = sum(v["lexicalized"] for v in col_stats.values())
    overall_rate = pct(total_lex, total_ref)

    col_summary_rows.append({
        "column": "OVERALL",
        "references": total_ref,
        "lexicalized": total_lex,
        "not_lexicalized": total_ref - total_lex,
        "lexicalization_rate_pct": overall_rate,
    })

    print(f"\n{'Column':<12} {'References':>10} {'Lexicalized':>12} {'Not lex.':>10} {'Rate%':>8}")
    print("-" * 56)
    for row in col_summary_rows:
        print(f"{row['column']:<12} {row['references']:>10} "
              f"{row['lexicalized']:>12} {row['not_lexicalized']:>10} "
              f"{row['lexicalization_rate_pct']:>8.1f}")

    return {
        "col_rows": col_rows,
        "col_summary_rows": col_summary_rows,
        "col_stats": col_stats,
        "total_ref": total_ref,
        "total_lex": total_lex,
        "overall_rate": overall_rate,
    }


def stage4_operation_lexicalization(corpus: list[dict]) -> dict:
    """
    For each utterance, detect SQL operations in gold SQL and check
    whether the utterance contains an explicit NL cue.
    """
    print("\n" + "="*60)
    print("STAGE 4.3: OPERATION LEXICALIZATION")
    print("="*60)

    op_rows: list[dict] = []
    ops_list = ["COUNT", "DISTINCT", "SUM", "MIN", "MAX", "GROUP BY",
                "Temporal", "Negation", "Set operation"]

    for item in corpus:
        sid = item["utterance_id"]
        gid = item["group_id"]
        sql = item["gold_sql"]
        q   = item["question"]

        try:
            ops_present = detect_sql_operations(sql)
        except Exception as e:
            print(f"[WARN] Operation detection failed for {sid}: {e}")
            ops_present = {op: False for op in ops_list}

        for op in ops_list:
            present = ops_present.get(op, False)
            if not present:
                has_cue = None  # NA
                status = "NA"
            else:
                has_cue = utterance_has_nl_cue(q, op)
                status = "explicit_cue" if has_cue else "no_cue"
            op_rows.append({
                "utterance_id": sid,
                "group_id": gid,
                "sql_operation": op,
                "operation_present_in_sql": present,
                "has_nl_cue": has_cue,   # None = NA
                "status": status,
            })

    # Aggregate per operation
    op_summary_rows = []
    op_stats: dict[str, dict] = {}
    for op in ops_list:
        op_data = [r for r in op_rows if r["sql_operation"] == op]
        present_data = [r for r in op_data if r["operation_present_in_sql"]]
        n_occ = len(present_data)
        n_cue = sum(1 for r in present_data if r["has_nl_cue"] is True)
        n_no  = n_occ - n_cue
        rate  = pct(n_cue, n_occ)
        op_stats[op] = {
            "sql_operation": op,
            "occurrences": n_occ,
            "explicit_cue": n_cue,
            "no_explicit_cue": n_no,
            "lexicalization_rate_pct": rate,
        }
        op_summary_rows.append(op_stats[op])

    # Overall
    total_occ = sum(v["occurrences"] for v in op_stats.values())
    total_cue = sum(v["explicit_cue"] for v in op_stats.values())
    overall_op_rate = pct(total_cue, total_occ)
    op_summary_rows.append({
        "sql_operation": "OVERALL",
        "occurrences": total_occ,
        "explicit_cue": total_cue,
        "no_explicit_cue": total_occ - total_cue,
        "lexicalization_rate_pct": overall_op_rate,
    })

    print(f"\n{'SQL Operation':<16} {'Occurrences':>12} {'Expl.cue':>10} {'No cue':>8} {'Rate%':>8}")
    print("-" * 60)
    for row in op_summary_rows:
        print(f"{row['sql_operation']:<16} {row['occurrences']:>12} "
              f"{row['explicit_cue']:>10} {row['no_explicit_cue']:>8} "
              f"{row['lexicalization_rate_pct']:>8.1f}")

    return {
        "op_rows": op_rows,
        "op_summary_rows": op_summary_rows,
        "op_stats": op_stats,
        "total_occ": total_occ,
        "total_cue": total_cue,
        "overall_op_rate": overall_op_rate,
    }


# ---------------------------------------------------------------------------
# 13. COMBINED UTTERANCE-LEVEL FILE
# ---------------------------------------------------------------------------

def build_utterance_level(
    corpus: list[dict],
    val_summary: dict[str, dict],
    col_rows: list[dict],
    op_rows: list[dict],
) -> list[dict]:
    """
    Build one row per utterance merging all analysis dimensions.
    Aggregates column and operation flags as comma-separated lists.
    """
    col_by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in col_rows:
        col_by_uid[r["utterance_id"]].append(r)

    op_by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in op_rows:
        op_by_uid[r["utterance_id"]].append(r)

    rows = []
    for item in corpus:
        sid = item["utterance_id"]
        v = val_summary.get(sid, {})
        # Column-level
        cols_ref = [r["column"] for r in col_by_uid[sid] if r["sql_references_column"]]
        cols_lex = [r["column"] for r in col_by_uid[sid]
                    if r["sql_references_column"] and r["column_lexicalized"] is True]
        # Operation-level
        ops_present = [r["sql_operation"] for r in op_by_uid[sid] if r["operation_present_in_sql"]]
        ops_cue     = [r["sql_operation"] for r in op_by_uid[sid]
                       if r["operation_present_in_sql"] and r["has_nl_cue"] is True]

        rows.append({
            "utterance_id": sid,
            "group_id": item["group_id"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
            "u_class": item["u_class"],
            "spider_class": item["spider_class"],
            # Value lexicalization
            "val_n_literals": v.get("n_literals", 0),
            "val_n_lexicalized": v.get("n_lexicalized", 0),
            "val_applicable": v.get("has_literals", False),
            # Column lexicalization
            "cols_referenced": ";".join(cols_ref),
            "cols_lexicalized": ";".join(cols_lex),
            "n_cols_referenced": len(cols_ref),
            "n_cols_lexicalized": len(cols_lex),
            # Operation lexicalization
            "ops_present": ";".join(ops_present),
            "ops_with_cue": ";".join(ops_cue),
            "n_ops_present": len(ops_present),
            "n_ops_with_cue": len(ops_cue),
        })
    return rows


# ---------------------------------------------------------------------------
# 14. THESIS-READY SUMMARY
# ---------------------------------------------------------------------------

def print_thesis_summary(
    val1: dict, lex2: dict, jac3: dict, val4: dict, col4: dict, op4: dict
) -> None:
    """Print a concise, factual thesis-ready summary."""
    print("\n" + "="*70)
    print("THESIS-READY SUMMARY")
    print("="*70)

    print("""
## Dataset Validation
""")
    print(f"The English evaluation corpus contains {val1['n_utterances']} utterances "
          f"organized into {val1['n_groups']} paraphrase groups.")
    print(f"Group-size statistics: mean = {val1['group_size_mean']:.2f}, "
          f"median = {val1['group_size_median']}, "
          f"min = {val1['group_size_min']}, "
          f"max = {val1['group_size_max']}.")
    if val1['n_inconsistent_groups'] == 0:
        print(f"All {val1['n_groups']} paraphrase groups contain a single "
              f"normalized gold SQL query (zero inconsistencies).")
    else:
        print(f"[REPORT] {val1['n_inconsistent_groups']} groups contain more than "
              f"one distinct gold SQL query.")

    print("""
## Lexical Diversity
""")
    ts = lex2["ttr_stats"]
    ms = lex2["mtld_stats"]
    print(f"Corpus-level TTR = {lex2['corpus_ttr']:.4f} "
          f"(types = {lex2['n_types']}, tokens = {lex2['n_tokens']}).")
    print(f"Corpus-level MTLD = {lex2['corpus_mtld']:.4f} "
          f"(threshold = {MTLD_THRESHOLD}).")
    print(f"Group-level TTR:  mean = {ts['mean']:.4f}, SD = {ts['sd']:.4f}, "
          f"median = {ts['median']:.4f}, range = [{ts['min']:.4f}, {ts['max']:.4f}].")
    print(f"Group-level MTLD: mean = {ms['mean']:.4f}, SD = {ms['sd']:.4f}, "
          f"median = {ms['median']:.4f}, range = [{ms['min']:.4f}, {ms['max']:.4f}].")

    print("""
## Surface Paraphrase Similarity
""")
    js = jac3["jac_stats"]
    print(f"Anchor–variant Jaccard similarity computed over {js['n']} pairs.")
    print(f"Mean Jaccard = {js['mean']:.4f} (SD = {js['sd']:.4f}), "
          f"median = {js['median']:.4f}, "
          f"Q1 = {js['q1']:.4f}, Q3 = {js['q3']:.4f}, "
          f"range = [{js['min']:.4f}, {js['max']:.4f}].")

    print("""
## SQL-Grounded Categorization
""")
    # Value
    vr = val4
    print(f"Value lexicalization: {vr['total_lex']} of {vr['total_literals']} "
          f"literal instances lexicalized "
          f"({pct(vr['total_lex'], vr['total_literals']):.1f}%) "
          f"across {vr['n_applicable_utts']} applicable utterances.")
    # Column
    cr = col4
    print(f"\nColumn lexicalization: {cr['total_lex']} of {cr['total_ref']} "
          f"column references lexicalized (overall rate = {cr['overall_rate']:.1f}%).")
    for row in cr["col_summary_rows"][:-1]:  # exclude OVERALL row
        print(f"  {row['column']:<10}: {row['lexicalized']}/{row['references']} "
              f"({row['lexicalization_rate_pct']:.1f}%)")
    # Operation
    orr = op4
    print(f"\nOperation lexicalization: {orr['total_cue']} of {orr['total_occ']} "
          f"operation occurrences have an explicit NL cue "
          f"(overall rate = {orr['overall_op_rate']:.1f}%).")
    for row in orr["op_summary_rows"][:-1]:
        if row["occurrences"] > 0:
            print(f"  {row['sql_operation']:<14}: {row['explicit_cue']}/{row['occurrences']} "
                  f"({row['lexicalization_rate_pct']:.1f}%)")


# ---------------------------------------------------------------------------
# 15. MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("="*70)
    print("TEXT2SQL4PM CORPUS ANALYSIS PIPELINE")
    print("="*70)
    print(f"Base directory  : {BASE_DIR}")
    print(f"dev.json        : {DEV_JSON}")
    print(f"gold.txt        : {GOLD_TXT}")
    print(f"text2sql4pm.tsv : {TSV_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    dev  = load_dev_json(DEV_JSON)
    gold = load_gold_txt(GOLD_TXT)
    tsv  = load_tsv(TSV_PATH)
    corpus = build_corpus(dev, gold, tsv)

    # ---- STAGE 1 ----
    s1 = stage1_validation(corpus)
    groups = s1["groups"]

    write_csv(OUTPUT_DIR / "dataset_validation.csv",
              s1["validation_rows"])

    # ---- STAGE 2 ----
    s2 = stage2_lexical_diversity(corpus, groups)

    write_csv(OUTPUT_DIR / "corpus_lexical_summary.csv",
              s2["corpus_summary"])
    write_csv(OUTPUT_DIR / "group_lexical_metrics.csv",
              s2["group_lex_rows"])

    # ---- STAGE 3 ----
    s3 = stage3_jaccard(groups)

    write_csv(OUTPUT_DIR / "jaccard_anchor_variant.csv",
              s3["pair_rows"],
              fieldnames=["group_id", "anchor_id", "variant_id",
                          "anchor_text", "variant_text", "jaccard_similarity"])

    jac_summary_rows = [{"metric": k, "value": v}
                        for k, v in s3["jac_stats"].items()]
    write_csv(OUTPUT_DIR / "jaccard_summary.csv", jac_summary_rows)

    # Group-mean Jaccard merged into pair_rows — also save separately
    group_jac_csv = []
    for r in s3["group_mean_rows"]:
        group_jac_csv.append(r)
    write_csv(OUTPUT_DIR / "jaccard_group_means.csv", group_jac_csv)

    # ---- STAGE 4.1 ----
    s4v = stage4_value_lexicalization(corpus)

    write_csv(OUTPUT_DIR / "value_lexicalization.csv",
              s4v["value_rows"],
              fieldnames=["utterance_id", "group_id", "gold_sql",
                          "literal_value", "value_type", "value_lexicalized"])
    write_csv(OUTPUT_DIR / "value_lexicalization_summary.csv",
              s4v["summary_rows"])

    # ---- STAGE 4.2 ----
    s4c = stage4_column_lexicalization(corpus)

    write_csv(OUTPUT_DIR / "column_lexicalization.csv",
              s4c["col_rows"],
              fieldnames=["utterance_id", "group_id", "column",
                          "sql_references_column", "column_lexicalized", "status"])
    write_csv(OUTPUT_DIR / "column_lexicalization_summary.csv",
              s4c["col_summary_rows"])

    # ---- STAGE 4.3 ----
    s4o = stage4_operation_lexicalization(corpus)

    write_csv(OUTPUT_DIR / "operation_lexicalization.csv",
              s4o["op_rows"],
              fieldnames=["utterance_id", "group_id", "sql_operation",
                          "operation_present_in_sql", "has_nl_cue", "status"])
    write_csv(OUTPUT_DIR / "operation_lexicalization_summary.csv",
              s4o["op_summary_rows"])

    # ---- Combined utterance-level file ----
    utt_rows = build_utterance_level(
        corpus,
        s4v["utterance_val_summary"],
        s4c["col_rows"],
        s4o["op_rows"],
    )
    write_csv(OUTPUT_DIR / "corpus_analysis_utterance_level.csv", utt_rows)

    # ---- Thesis summary ----
    print_thesis_summary(s1, s2, s3, s4v, s4c, s4o)

    print("\n" + "="*70)
    print(f"All outputs saved in: {OUTPUT_DIR}")
    print("="*70)

    # ---- Final sanity check ----
    print("\n--- SANITY CHECKS ---")
    print(f"English corpus utterances : {s1['n_utterances']}  "
          f"(expected {EXPECTED_UTTERANCES}) -> {'PASS' if s1['utterance_count_ok'] else 'FAIL'}")
    print(f"Paraphrase groups         : {s1['n_groups']}  "
          f"(expected {EXPECTED_GROUPS}) -> {'PASS' if s1['group_count_ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
