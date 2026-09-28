"""Pipeline orchestrator.

The runner wires the modules together and runs the main loop. It knows
nothing about backends, prompt strategies, or evaluators in detail — it
operates only through the abstract interfaces, so adding a new backend or
strategy never requires touching this file.

Three top-level entry points correspond to ``--execution``:

  - ``run_generate(cfg)``       — produce predictions only (no scoring).
  - ``run_evaluate(cfg)``       — score an existing predictions.jsonl.
  - ``run_full(cfg)``           — generate AND evaluate inline (default).
"""
from __future__ import annotations

import time
import traceback
from typing import Optional

from tqdm import tqdm

from .adapters import (
    AuthError,
    ModelAdapter,
    RateLimitError,
    ServerError,
    TimeoutError_,
    build_adapter,
)
from .config import RunConfig
from .data_loader import DataLoader, Sample
from .eval import ExactMatch, ExecutionAccuracy
from .executor import SQLExecutor
from .postprocess import clean_sql
from .prompts import PromptStrategy, build_strategy
from .writer import ResultWriter, SampleRecord


# ---------------------------------------------------------------------------
def _wants_em(cfg: RunConfig) -> bool:
    return cfg.metric in ("EM", "both")


def _wants_ex(cfg: RunConfig) -> bool:
    return cfg.metric in ("EX", "both")


def _completion_prefix(strategy: PromptStrategy) -> str:
    """Read the strategy's declared completion prefix (empty for chat-style)."""
    return getattr(strategy, "completion_prefix", "") or ""


# ---------------------------------------------------------------------------
def run_full(cfg: RunConfig) -> dict:
    """Generate predictions and score them in a single pass — the default flow."""
    loader = DataLoader(
        questions_path=cfg.questions_path,
        dev_path=cfg.dev_path,
        gold_path=cfg.gold_path,
        hardness_tsv_path=cfg.hardness_tsv_path,
    ).load()

    strategy = build_strategy(cfg)
    adapter: ModelAdapter = build_adapter(cfg)
    sql_executor = SQLExecutor(cfg.db_path, timeout_seconds=cfg.sql_exec_timeout)

    em_evaluator = ExactMatch() if _wants_em(cfg) else None
    ex_evaluator = ExecutionAccuracy(sql_executor) if _wants_ex(cfg) else None

    writer = ResultWriter(cfg)
    writer.write_initial_meta(dataset_path=cfg.gold_path)

    samples = list(loader.samples(limit=cfg.limit))
    total = len(samples)
    failed = 0

    print(f"\n→ Experiment: {writer.dir}")
    print(f"  source={cfg.source}  model={cfg.model}  prompt={cfg.prompt}  metric={cfg.metric}")
    print(f"  samples={total}  resume={cfg.resume} (completed={len(writer.completed_ids)})\n")

    try:
        for sample in tqdm(samples, desc="generate", unit="q"):
            if cfg.resume and sample.id in writer.completed_ids:
                continue

            rec = _process_one(
                sample=sample,
                strategy=strategy,
                adapter=adapter,
                sql_executor=sql_executor,
                em_evaluator=em_evaluator,
                ex_evaluator=ex_evaluator,
                cfg=cfg,
            )
            if rec.error:
                failed += 1
                if cfg.on_error == "abort":
                    writer.write_sample(rec)
                    raise RuntimeError(f"Aborting on error at {sample.id}: {rec.error}")

            writer.write_sample(rec)

    finally:
        writer.finalize_meta(samples_total=total, samples_failed=failed)
        writer.close()
        adapter.close()
        sql_executor.close()

    # Console summary.
    em = writer.em_sum / writer.em_n if writer.em_n else None
    ex = writer.ex_sum / writer.ex_n if writer.ex_n else None
    print()
    if em is not None:
        print(f"  EM: {em:.4f}  ({writer.em_sum}/{writer.em_n})")
    if ex is not None:
        print(f"  EX: {ex:.4f}  ({writer.ex_sum}/{writer.ex_n})")
    print(f"  failed: {failed}/{total}")
    print(f"  output: {writer.dir}\n")

    return {
        "experiment_dir": str(writer.dir),
        "samples_total": total,
        "samples_failed": failed,
        "em": em,
        "ex": ex,
    }


# ---------------------------------------------------------------------------
def _process_one(
    *,
    sample: Sample,
    strategy: PromptStrategy,
    adapter: ModelAdapter,
    sql_executor: SQLExecutor,
    em_evaluator: Optional[ExactMatch],
    ex_evaluator: Optional[ExecutionAccuracy],
    cfg: RunConfig,
) -> SampleRecord:
    """Run prompt → model → postprocess → execute → score on one sample."""
    rec = SampleRecord(
        id=sample.id,
        question=sample.question,
        db_id=sample.db_id,
        hardness=sample.hardness,
        prompt_messages=[],
        raw_response="",
        predicted_sql="",
        gold_sql=sample.gold_sql,
    )

    # 1. Build prompt.
    try:
        messages = strategy.build(sample)
        rec.prompt_messages = [m.to_dict() for m in messages]
    except Exception as e:
        rec.error = f"prompt_build: {e}\n{traceback.format_exc()}"
        return rec

    # 2. Call model with typed retry / skip policy.
    start = time.perf_counter()
    try:
        result = adapter.generate(
            messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    except AuthError as e:
        # Auth errors are always fatal: every subsequent call will fail the same way.
        rec.error = f"auth: {e}"
        raise
    except (TimeoutError_, RateLimitError, ServerError) as e:
        # Recoverable per the adapter's internal retry; if we still get here,
        # the retries are exhausted. Skip the sample with a recorded error.
        rec.error = f"{type(e).__name__}: {e}"
        rec.latency_ms = (time.perf_counter() - start) * 1000.0
        return rec
    except Exception as e:
        rec.error = f"model_call: {e}"
        rec.latency_ms = (time.perf_counter() - start) * 1000.0
        return rec

    rec.raw_response = result.text
    rec.latency_ms = (time.perf_counter() - start) * 1000.0
    rec.prompt_tokens = result.prompt_tokens
    rec.completion_tokens = result.completion_tokens
    rec.total_tokens = result.total_tokens
    rec.prompt_n = result.prompt_n
    rec.prompt_ms = result.prompt_ms
    rec.prompt_per_token_ms = result.prompt_per_token_ms
    rec.prompt_per_second = result.prompt_per_second
    rec.predicted_n = result.predicted_n
    rec.predicted_ms = result.predicted_ms
    rec.predicted_per_token_ms = result.predicted_per_token_ms
    rec.predicted_per_second = result.predicted_per_second

    # 3. Postprocess.
    rec.predicted_sql = clean_sql(
        result.text,
        cot=(cfg.prompt == "cot"),
        completion_prefix=_completion_prefix(strategy),
    )

    # 4. Execute predicted SQL once; EX evaluator reuses the result.
    pred_exec = sql_executor.execute(rec.predicted_sql) if ex_evaluator else None
    if pred_exec is not None:
        rec.pred_exec_ok = pred_exec.ok
        rec.pred_exec_error = pred_exec.error

    # 5. Score.
    if em_evaluator is not None:
        em = em_evaluator.evaluate(rec.predicted_sql, sample.gold_sql)
        rec.em = em.score
        rec.em_detail = em.detail

    if ex_evaluator is not None:
        gold_exec = sql_executor.execute(sample.gold_sql)
        ex = ex_evaluator.evaluate(
            rec.predicted_sql,
            sample.gold_sql,
            pred_exec=pred_exec,
            gold_exec=gold_exec,
        )
        rec.ex = ex.score
        rec.ex_detail = ex.detail

    return rec


# ---------------------------------------------------------------------------
def run_generate(cfg: RunConfig) -> dict:
    """Generate-only path: same as run_full but with metric scoring disabled."""
    # We achieve this by forcing both evaluators off; everything else identical.
    overridden = RunConfig(**{**cfg.to_dict(), "metric": "none"})
    # The to_dict() helper redacts secrets — refill from the live cfg.
    overridden.username = cfg.username
    overridden.password = cfg.password
    overridden.hf_token = cfg.hf_token
    return run_full(overridden)


def run_evaluate(cfg: RunConfig) -> dict:
    """Score an existing predictions.jsonl without re-calling the model.

    Reads ``predictions.jsonl`` from the experiment dir, re-runs the SQL
    executor + evaluators, and rewrites the scores TSVs. Useful when you
    want to add a metric (e.g. EX) to a run that was generated EM-only.
    """
    import json

    from .naming import experiment_dir, scores_filename

    out_dir = experiment_dir(cfg)
    preds_path = out_dir / "predictions.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(f"No predictions to evaluate at {preds_path}")

    sql_executor = SQLExecutor(cfg.db_path, timeout_seconds=cfg.sql_exec_timeout)
    em_evaluator = ExactMatch() if _wants_em(cfg) else None
    ex_evaluator = ExecutionAccuracy(sql_executor) if _wants_ex(cfg) else None

    em_rows: list[list] = []
    ex_rows: list[list] = []
    em_sum = em_n = ex_sum = ex_n = 0

    with preds_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pred = obj.get("predicted_sql") or ""
            gold = obj.get("gold_sql") or ""
            sid = obj["id"]
            hardness = obj.get("hardness", "no_hardness")

            if em_evaluator:
                r = em_evaluator.evaluate(pred, gold)
                em_rows.append([sid, pred, gold, hardness, r.score])
                em_sum += r.score
                em_n += 1
            if ex_evaluator:
                r = ex_evaluator.evaluate(pred, gold)
                ex_rows.append([sid, pred, gold, hardness, r.score])
                ex_sum += r.score
                ex_n += 1

    eval_dir = out_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if em_evaluator:
        _write_tsv(eval_dir / scores_filename(cfg, "EM"), em_rows)
    if ex_evaluator:
        _write_tsv(eval_dir / scores_filename(cfg, "EX"), ex_rows)

    em_score = (em_sum / em_n) if em_n else None
    ex_score = (ex_sum / ex_n) if ex_n else None
    print()
    if em_score is not None:
        print(f"  EM: {em_score:.4f}  ({em_sum}/{em_n})")
    if ex_score is not None:
        print(f"  EX: {ex_score:.4f}  ({ex_sum}/{ex_n})")
    print(f"  output: {out_dir}\n")

    sql_executor.close()
    return {
        "experiment_dir": str(out_dir),
        "em": em_score,
        "ex": ex_score,
    }


def _write_tsv(path, rows) -> None:
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\r\n")
        for r in rows:
            w.writerow(r)
