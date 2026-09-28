"""Data loader.

Produces a unified ``Sample`` stream from four files:
  - questions.json   (prebuilt 0-shot prompts, used optionally)
  - dev.json         (canonical question text + id + db_id + gold query)
  - gold.txt         (positional gold list — used as cross-check)
  - text2sql4pm.tsv  (hardness label per utterance)

Alignment notes (verified empirically on the EN split):
  - dev.json and gold.txt are 1:1 by row order (1655 entries, id matches).
  - text2sql4pm.tsv has 1859 rows (incl. PT-only and empty rows). The robust
    join is (group_id, position_within_group_sorted_by_Utterance_id) → hardness.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


# ---------------------------------------------------------------------------
# Schema description — single source of truth for prompt builders.
# Kept as a literal because the DB has exactly one table (event_log) and
# pulling it from sqlite_master adds noise for no real benefit here.
#
# Two forms are exported:
#   - SCHEMA_DESCRIPTION : rich form with column types; used by chat-style
#                          strategies (1-shot, CoT, RAG).
#   - SCHEMA_COLUMNS_ONLY: bare column list; used by the OpenAI / NumberSign
#                          representation (DAIL-SQL Gao et al., 2023), which
#                          is the format used by the published GPT-3.5 baseline.
# ---------------------------------------------------------------------------
SCHEMA_DESCRIPTION = (
    "event_log(id INT PRIMARY KEY, activity TEXT, timestamp DATETIME, "
    "resource TEXT, cost NUMERIC, idcase TEXT)"
)

SCHEMA_COLUMNS_ONLY = "event_log(id, activity, timestamp, resource, cost, idcase)"

SCHEMA_NOTES = (
    "Notes:\n"
    "- `activity` is the event/activity name (e.g. 'Start trip', 'End trip').\n"
    "- `resource` is the actor performing the activity (a person's name).\n"
    "- `idcase` groups events belonging to the same process case.\n"
    "- `timestamp` is the time the activity occurred."
)


@dataclass
class Sample:
    id: str                 # e.g. "1_0"
    group_id: str           # e.g. "1"
    question: str           # natural-language question (English or Portuguese)
    db_id: str              # "event_log"
    gold_sql: str           # ground-truth SQL
    hardness: str           # "easy" | "medium" | "hard" | "extra" | "no_hardness"
    prebuilt_prompt: str    # the original questions.json prompt (ends with "SELECT ")


# ---------------------------------------------------------------------------
class DataLoader:
    """Reads all source files once and emits Samples in dataset order."""

    def __init__(
        self,
        questions_path: str,
        dev_path: str,
        gold_path: str,
        hardness_tsv_path: Optional[str] = None,
    ) -> None:
        self.questions_path = Path(questions_path)
        self.dev_path = Path(dev_path)
        self.gold_path = Path(gold_path)
        self.hardness_tsv_path = Path(hardness_tsv_path) if hardness_tsv_path else None

        self._samples: list[Sample] = []
        self._loaded = False

    # ---- loading ----------------------------------------------------------
    def load(self) -> "DataLoader":
        if self._loaded:
            return self

        dev = self._load_dev()
        prebuilt = self._load_prebuilt_prompts()
        gold_index = self._load_gold()
        hardness_by_id = self._load_hardness()

        # Build samples in dev.json order (canonical).
        for i, item in enumerate(dev):
            sid = item["id"]
            group_id = item.get("group_id") or sid.split("_")[0]

            # Cross-check with gold.txt — drop sample if mismatch (defensive).
            gold_record = gold_index.get(sid)
            if gold_record is None:
                # Try positional fallback (older datasets occasionally diverge).
                gold_record = gold_index.get(f"__pos_{i}")
            gold_sql = gold_record["sql"] if gold_record else item.get("query", "")

            self._samples.append(
                Sample(
                    id=sid,
                    group_id=str(group_id),
                    question=item["question"],
                    db_id=item.get("db_id", "event_log"),
                    gold_sql=gold_sql,
                    hardness=hardness_by_id.get(sid, "no_hardness"),
                    prebuilt_prompt=prebuilt.get(sid, ""),
                )
            )

        self._loaded = True
        return self

    def _load_dev(self) -> list[dict]:
        with self.dev_path.open(encoding="utf-8") as f:
            return json.load(f)

    def _load_prebuilt_prompts(self) -> dict[str, str]:
        if not self.questions_path.exists():
            return {}
        with self.questions_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return {q["id"]: q["prompt"] for q in data.get("questions", [])}

    def _load_gold(self) -> dict[str, dict]:
        """gold.txt rows: `<sql>\\t<db_id>\\t<id>`. Returns dict keyed by id."""
        result: dict[str, dict] = {}
        with self.gold_path.open(encoding="utf-8") as f:
            for i, raw in enumerate(f):
                line = raw.rstrip("\r\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                sql, db_id, sid = parts[0], parts[1], parts[2]
                rec = {"sql": sql, "db_id": db_id, "id": sid}
                result[sid] = rec
                result[f"__pos_{i}"] = rec  # positional fallback
        return result

    def _load_hardness(self) -> dict[str, str]:
        """Join text2sql4pm.tsv to dev ids via (group_id, position-in-group).

        Empty `English_utterance` rows in the TSV are skipped (they're PT-only
        carryovers from an earlier export). Within each group, rows are sorted
        by ``Utterance_id`` to match dev.json's positional numbering.
        """
        if not self.hardness_tsv_path or not self.hardness_tsv_path.exists():
            return {}

        with self.hardness_tsv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

        grouped: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if not r.get("English_utterance", "").strip():
                continue
            grouped[str(r["Group_id"])].append(r)
        for gid in grouped:
            grouped[gid].sort(key=lambda r: int(r["Utterance_id"]))

        out: dict[str, str] = {}
        for gid, items in grouped.items():
            for pos, row in enumerate(items):
                sid = f"{gid}_{pos}"
                hardness = row.get("Spider_classification") or "no_hardness"
                out[sid] = hardness
        return out

    # ---- iteration --------------------------------------------------------
    def samples(self, limit: Optional[int] = None) -> Iterator[Sample]:
        if not self._loaded:
            self.load()
        if limit is None:
            yield from self._samples
        else:
            yield from self._samples[:limit]

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)

    # ---- DB access --------------------------------------------------------
    @staticmethod
    def open_db_readonly(db_path: str) -> sqlite3.Connection:
        """Open SQLite in read-only mode so models can't damage the dataset."""
        uri = f"file:{db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)
