"""Result writer.

Owns the on-disk layout. Four files per run, all opened at start and
flushed after every sample so a kill mid-run leaves recoverable state.

Files written
-------------
1. ``RESULTS_MODEL-<safe-name>.txt``
     One predicted SQL per line, gold-aligned. Backwards-compatible with
     the existing ``scripts/results_analysis.py``.

2. ``predictions.jsonl``
     Rich per-sample log used for ``--resume`` and post-hoc analysis.

3. ``evaluations/scores_opr_<slug>_{EM|EX}.tsv``
     The canonical evaluation TSV format: ``id\\tpred\\tgold\\thardness\\tscore``.
     Same shape ``loader_results.py`` expects.

4. ``run_meta.json``
     Full config snapshot + timestamps + dataset hash. Written at start
     (partial), updated again at end.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional

from .config import RunConfig
from .naming import experiment_dir, experiment_name, results_filename, scores_filename


@dataclass
class SampleRecord:
    """Everything we know about one sample after running it through the pipeline."""

    id: str
    question: str
    db_id: str
    hardness: str
    prompt_messages: list[dict]
    raw_response: str
    predicted_sql: str
    gold_sql: str
    em: Optional[int] = None
    ex: Optional[int] = None
    em_detail: dict = field(default_factory=dict)
    ex_detail: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_n: Optional[int] = None
    prompt_ms: Optional[float] = None
    prompt_per_token_ms: Optional[float] = None
    prompt_per_second: Optional[float] = None
    predicted_n: Optional[int] = None
    predicted_ms: Optional[float] = None
    predicted_per_token_ms: Optional[float] = None
    predicted_per_second: Optional[float] = None
    pred_exec_ok: Optional[bool] = None
    pred_exec_error: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
class ResultWriter:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.dir = experiment_dir(cfg)
        self.eval_dir = self.dir / "evaluations"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.results_path = self.dir / results_filename(cfg)
        self.predictions_path = self.dir / "predictions.jsonl"
        self.meta_path = self.dir / "run_meta.json"
        self.em_path = self.eval_dir / scores_filename(cfg, "EM")
        self.ex_path = self.eval_dir / scores_filename(cfg, "EX")

        # Resume support: load completed ids from existing predictions.jsonl.
        self.completed_ids: set[str] = set()
        if cfg.resume:
            self.completed_ids = self._read_completed_ids()

        # File handles opened lazily on first write (so dry-run doesn't litter).
        self._f_results: Optional[IO[str]] = None
        self._f_preds: Optional[IO[str]] = None
        self._f_em: Optional[IO[str]] = None
        self._f_ex: Optional[IO[str]] = None

        # Counters used for the final meta update.
        self.n_written = 0
        self.em_sum = 0
        self.ex_sum = 0
        self.em_n = 0
        self.ex_n = 0
        self.total_tokens = 0
        self.token_query_n = 0
        self.total_prompt_n = 0
        self.total_prompt_ms = 0.0
        self.total_predicted_n = 0
        self.total_predicted_ms = 0.0
        self.timing_query_n = 0

    # ---- meta -------------------------------------------------------------
    def write_initial_meta(self, dataset_path: str) -> None:
        meta = {
            "experiment_name": experiment_name(self.cfg),
            "config": self.cfg.to_dict(),
            "started_at": dt.datetime.utcnow().isoformat() + "Z",
            "finished_at": None,
            "git_sha": _git_sha(),
            "dataset_sha256": _file_sha256(dataset_path),
            "samples_total": None,
            "samples_failed": None,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def finalize_meta(self, samples_total: int, samples_failed: int) -> None:
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            meta = {}
        meta["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
        meta["samples_total"] = samples_total
        meta["samples_failed"] = samples_failed
        meta["em_score"] = (self.em_sum / self.em_n) if self.em_n else None
        meta["ex_score"] = (self.ex_sum / self.ex_n) if self.ex_n else None
        meta["total_tokens"] = self.total_tokens if self.token_query_n else None
        meta["tokens_per_query"] = (
            round(self.total_tokens / self.token_query_n, 2) if self.token_query_n else None
        )
        if self.timing_query_n:
            pn = self.total_prompt_n
            pms = self.total_prompt_ms
            dn = self.total_predicted_n
            dms = self.total_predicted_ms
            meta["prompt_n"] = pn
            meta["prompt_ms"] = round(pms, 3)
            meta["prompt_per_token_ms"] = round(pms / pn, 6) if pn else None
            meta["prompt_per_second"] = round(pn / (pms / 1000), 2) if pms else None
            meta["predicted_n"] = dn
            meta["predicted_ms"] = round(dms, 3)
            meta["predicted_per_token_ms"] = round(dms / dn, 6) if dn else None
            meta["predicted_per_second"] = round(dn / (dms / 1000), 2) if dms else None
        else:
            for key in ("prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
                        "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"):
                meta[key] = None
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- streaming writes -------------------------------------------------
    def _open_outputs(self) -> None:
        mode = "a" if (self.cfg.resume and self.results_path.exists()) else "w"
        self._f_results = self.results_path.open(mode, encoding="utf-8", newline="\n")
        self._f_preds = self.predictions_path.open(mode, encoding="utf-8")
        self._f_em = self.em_path.open(mode, encoding="utf-8", newline="")
        self._f_ex = self.ex_path.open(mode, encoding="utf-8", newline="")

    def write_sample(self, rec: SampleRecord) -> None:
        if self._f_results is None:
            self._open_outputs()

        # 1) Legacy one-SQL-per-line file (gold-aligned, no id column).
        self._f_results.write((rec.predicted_sql or "") + "\n")
        self._f_results.flush()

        # 2) Rich JSONL log.
        self._f_preds.write(json.dumps(_serializable(rec), ensure_ascii=False) + "\n")
        self._f_preds.flush()

        # 3) Score TSVs. We always write both (with whichever score is known).
        if rec.em is not None:
            self._write_tsv_row(self._f_em, rec, rec.em)
            self.em_sum += rec.em
            self.em_n += 1
        if rec.ex is not None:
            self._write_tsv_row(self._f_ex, rec, rec.ex)
            self.ex_sum += rec.ex
            self.ex_n += 1
        if rec.total_tokens is not None:
            self.total_tokens += rec.total_tokens
            self.token_query_n += 1
        if rec.prompt_n is not None and rec.predicted_n is not None:
            self.total_prompt_n += rec.prompt_n
            self.total_prompt_ms += rec.prompt_ms or 0.0
            self.total_predicted_n += rec.predicted_n
            self.total_predicted_ms += rec.predicted_ms or 0.0
            self.timing_query_n += 1

        self.n_written += 1

    @staticmethod
    def _write_tsv_row(f: IO[str], rec: SampleRecord, score: int) -> None:
        w = csv.writer(f, delimiter="\t", lineterminator="\r\n")
        w.writerow([rec.id, rec.predicted_sql, rec.gold_sql, rec.hardness, score])
        f.flush()

    # ---- resume helpers ---------------------------------------------------
    def _read_completed_ids(self) -> set[str]:
        if not self.predictions_path.exists():
            return set()
        ids: set[str] = set()
        with self.predictions_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        ids.add(obj["id"])
                except json.JSONDecodeError:
                    continue
        return ids

    # ---- cleanup ----------------------------------------------------------
    def close(self) -> None:
        for f in (self._f_results, self._f_preds, self._f_em, self._f_ex):
            if f is not None:
                f.close()
        self._f_results = self._f_preds = self._f_em = self._f_ex = None


# ---------------------------------------------------------------------------
def _serializable(rec: SampleRecord) -> dict:
    return {
        "id": rec.id,
        "question": rec.question,
        "db_id": rec.db_id,
        "hardness": rec.hardness,
        "prompt_messages": rec.prompt_messages,
        "raw_response": rec.raw_response,
        "predicted_sql": rec.predicted_sql,
        "gold_sql": rec.gold_sql,
        "em": rec.em,
        "ex": rec.ex,
        "em_detail": rec.em_detail,
        "ex_detail": rec.ex_detail,
        "latency_ms": rec.latency_ms,
        "prompt_tokens": rec.prompt_tokens,
        "completion_tokens": rec.completion_tokens,
        "total_tokens": rec.total_tokens,
        "prompt_n": rec.prompt_n,
        "prompt_ms": rec.prompt_ms,
        "prompt_per_token_ms": rec.prompt_per_token_ms,
        "prompt_per_second": rec.prompt_per_second,
        "predicted_n": rec.predicted_n,
        "predicted_ms": rec.predicted_ms,
        "predicted_per_token_ms": rec.predicted_per_token_ms,
        "predicted_per_second": rec.predicted_per_second,
        "pred_exec_ok": rec.pred_exec_ok,
        "pred_exec_error": rec.pred_exec_error,
        "error": rec.error,
    }


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def _file_sha256(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
