"""Evaluator base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..executor import ExecResult


@dataclass
class EvalResult:
    score: int                    # 0 or 1
    detail: dict = field(default_factory=dict)


class Evaluator(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        predicted_sql: str,
        gold_sql: str,
        pred_exec: Optional[ExecResult] = None,
        gold_exec: Optional[ExecResult] = None,
    ) -> EvalResult:
        raise NotImplementedError
