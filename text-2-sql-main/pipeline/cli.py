"""Command-line interface.

Maps argparse args onto a typed ``RunConfig`` and dispatches to the runner.
This module is the only place that knows about argparse — every other
module receives a ``RunConfig`` and works with that.
"""
from __future__ import annotations

import argparse
from typing import Optional

from .config import RunConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="t2s-bench",
        description="Text-to-SQL benchmarking pipeline (text2sql4pm dataset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- what to do -------------------------------------------------------
    p.add_argument(
        "--execution",
        choices=["generate", "evaluate", "result_analysis"],
        default="result_analysis",
        help="generate=produce predictions only; "
             "evaluate=score existing predictions; "
             "result_analysis=generate + score (default).",
    )
    p.add_argument("--language", choices=["english", "portuguese"], default="english")
    p.add_argument(
        "--perspective",
        default="sql",
        help="Stored in run_meta for downstream qualifier analysis (sql / process_mining / nlp).",
    )
    p.add_argument(
        "--metric",
        choices=["EM", "EX", "both"],
        default="EX",
        help="Which evaluation metric(s) to compute.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap number of samples evaluated.")

    # ---- prompt strategy --------------------------------------------------
    p.add_argument(
        "--prompt",
        choices=[
            "zeroshot",     # PureZeroShot   — plain instruction only
            "lexicon",      # LexiconGuided — instruction + domain lexicon
            "lf",           # LogicalFormGuided — instruction + logical-form guidance
            "lf-lite",      # LogicalFormLiteGuided — instruction + lf lite guidance
            "lexicon-lf",   # LexiconLogicalFormGuided — instruction + lexicon + lf
            "0-shot",       # alias for zeroshot
        ],
        default="zeroshot",
        help="Prompt strategy: zeroshot | lexicon | lf | lf-lite | lexicon-lf.",
    )
    p.add_argument("--n_shots", type=int, default=1)
    p.add_argument("--rag_top_k", type=int, default=3)
    p.add_argument(
        "--use_prebuilt_prompt",
        action="store_true",
        help="0-shot only: pass the verbatim questions.json prompt instead "
             "of building a fresh chat-formatted one.",
    )

    # ---- backend ----------------------------------------------------------
    p.add_argument("--source", choices=["local", "hf"], required=True,
                   help="Backend type.")
    p.add_argument("--model", required=True,
                   help="Model identifier (HF repo id, or display name for local API).")

    # local API
    p.add_argument("--api_url", default="http://10.204.18.32:8080/v1/chat/completions")
    p.add_argument("--username", default=None,
                   help="Defaults to $T2S_API_USERNAME.")
    p.add_argument("--password", default=None,
                   help="Defaults to $T2S_API_PASSWORD.")

    # HF
    p.add_argument("--hf_token", default=None,
                   help="Defaults to $HF_TOKEN or $HUGGINGFACE_TOKEN.")
    p.add_argument("--hf_mode", choices=["api", "local"], default="api",
                   help="api=InferenceClient over HTTP; local=transformers on this machine.")

    # ---- generation -------------------------------------------------------
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=2000)
    p.add_argument("--request_timeout", type=int, default=300)
    p.add_argument("--request_retries", type=int, default=3)

    # ---- robustness -------------------------------------------------------
    p.add_argument("--on_error", choices=["skip", "abort"], default="skip")
    p.add_argument("--sql_exec_timeout", type=int, default=5)
    p.add_argument("--resume", action="store_true",
                   help="Skip ids already present in predictions.jsonl.")

    # ---- output -----------------------------------------------------------
    p.add_argument("--results_root", default="results")
    p.add_argument("--data_root", default="data")
    p.add_argument("--run_tag", default=None,
                   help="Optional disambiguator appended to the experiment name.")

    return p


def parse_args(argv: Optional[list[str]] = None) -> RunConfig:
    args = build_parser().parse_args(argv)
    cfg = RunConfig(
        execution=args.execution,
        language=args.language,
        perspective=args.perspective,
        metric=args.metric,
        limit=args.limit,
        prompt=args.prompt,
        n_shots=args.n_shots,
        rag_top_k=args.rag_top_k,
        use_prebuilt_prompt=args.use_prebuilt_prompt,
        source=args.source,
        model=args.model,
        api_url=args.api_url,
        username=args.username,
        password=args.password,
        hf_token=args.hf_token,
        hf_mode=args.hf_mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        request_retries=args.request_retries,
        on_error=args.on_error,
        sql_exec_timeout=args.sql_exec_timeout,
        resume=args.resume,
        results_root=args.results_root,
        data_root=args.data_root,
        run_tag=args.run_tag,
    )
    return cfg
