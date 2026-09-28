"""Text-to-SQL benchmarking pipeline — entry point.

Usage examples
--------------
Local API (qwopus_glm_18b_healed_q4_k_m on the lab server), full 100-sample run with EX:

    python main.py \\
        --source local \\
        --api_url http://10.204.18.32:8080/v1/chat/completions \\
        --model qwopus_glm_18b_healed_q4_k_m \\
        --execution result_analysis \\
        --perspective sql \\
        --prompt 0-shot \\
        --metric EX \\
        --limit 100

HuggingFace Inference API, 10-sample CoT, both metrics:

    python main.py \\
        --source hf \\
        --model meta-llama/Llama-3-8B-Instruct \\
        --hf_token $HF_TOKEN \\
        --execution result_analysis \\
        --prompt cot \\
        --metric both \\
        --limit 10

Re-score an existing run (no model calls):

    python main.py --source local --model qwopus_glm_18b_healed_q4_k_m \\
        --execution evaluate --prompt 0-shot --metric both
"""
from __future__ import annotations

import sys

from pipeline.cli import parse_args
from pipeline.runner import run_evaluate, run_full, run_generate


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)

    if cfg.execution == "generate":
        run_generate(cfg)
    elif cfg.execution == "evaluate":
        run_evaluate(cfg)
    elif cfg.execution == "result_analysis":
        run_full(cfg)
    else:  # pragma: no cover — argparse guards the choices
        print(f"Unknown execution: {cfg.execution}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
