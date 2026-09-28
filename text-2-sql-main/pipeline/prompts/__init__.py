"""Prompt strategy registry.

To add a new strategy:
  1. Create a new file in this package implementing ``PromptStrategy``.
  2. Add an entry to ``STRATEGIES`` below.
  3. Add the name to the ``--prompt`` argparse choices in ``pipeline/cli.py``.
Nothing else changes.

Experiment strategies (experiment.py) embed the schema directly in their
system prompts, so they do not use ``schema_description`` / ``schema_notes``.
"""
from __future__ import annotations

from typing import Type

from ..config import RunConfig
from .base import ChatMessage, PromptStrategy
from .experiment import (
    LexiconGuided,
    LexiconLogicalFormGuided,
    LogicalFormGuided,
    LogicalFormLiteGuided,
    PureZeroShot,
)

STRATEGIES: dict[str, Type[PromptStrategy]] = {
    "zeroshot": PureZeroShot,
    "0-shot": PureZeroShot,  # backward-compatible alias
    "lexicon": LexiconGuided,
    "lf": LogicalFormGuided,
    "lf-lite": LogicalFormLiteGuided,
    "lexicon-lf": LexiconLogicalFormGuided,
}


def build_strategy(cfg: RunConfig) -> PromptStrategy:
    """Instantiate the strategy named by ``cfg.prompt``."""
    if cfg.prompt not in STRATEGIES:
        raise ValueError(
            f"Unknown prompt strategy '{cfg.prompt}'. "
            f"Known: {sorted(STRATEGIES)}"
        )

    cls = STRATEGIES[cfg.prompt]
    return cls(schema_description="", schema_notes="")


__all__ = [
    "PromptStrategy",
    "ChatMessage",
    "STRATEGIES",
    "build_strategy",
    "PureZeroShot",
    "LexiconGuided",
    "LogicalFormGuided",
    "LogicalFormLiteGuided",
    "LexiconLogicalFormGuided",
]

