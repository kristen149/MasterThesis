"""Experiment naming.

The slug must be:
  - deterministic (same config → same slug, so re-runs land in the same folder)
  - filesystem-safe (no slashes, spaces, or punctuation)
  - human-readable (so you can grep `results/english/` and know what's what)
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import RunConfig


def slugify(text: str) -> str:
    """Lowercase, collapse non-alphanumeric to `_`, trim leading/trailing `_`."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s or "unknown"


def source_tag(source: str) -> str:
    # 'oss' is the convention used in the spec for any open-source model
    # routed through either backend; keeping it stable so legacy comparisons work.
    if source == "local":
        return "oss"
    if source == "hf":
        return "hf"
    return slugify(source)


def short_model_slug(model_name: str) -> str:
    """Drop the org prefix from `Qwen/Qwen2.5-7B-Instruct` → `qwen2_5_7b_instruct`."""
    tail = model_name.split("/")[-1] if "/" in model_name else model_name
    return slugify(tail)


def experiment_name(cfg: RunConfig) -> str:
    src = source_tag(cfg.source)
    m_slug = short_model_slug(cfg.model)
    if m_slug.startswith(f"{src}_"):
        parts = [m_slug, cfg.prompt]
    else:
        parts = [src, m_slug, cfg.prompt]
    if cfg.run_tag:
        parts.append(slugify(cfg.run_tag))
    return "_".join(parts)


def experiment_dir(cfg: RunConfig) -> Path:
    return Path(cfg.results_root) / cfg.language / experiment_name(cfg)


def results_filename(cfg: RunConfig) -> str:
    """RESULTS_MODEL-<safe_model_name>.txt — kept compatible with legacy analysis."""
    safe = cfg.model.replace("/", "-")
    return f"RESULTS_MODEL-{safe}.txt"


def scores_filename(cfg: RunConfig, metric: str) -> str:
    """e.g. scores_opr_qwen2_5_7b_instruct_EM.tsv."""
    return f"scores_opr_{short_model_slug(cfg.model)}_{metric}.tsv"
