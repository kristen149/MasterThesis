"""Model adapter base class.

Both backends present the same single-method interface:

    adapter.generate(messages, **gen_kwargs) -> GenerationResult

so the runner is identical regardless of backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..prompts.base import ChatMessage


# ---------------------------------------------------------------------------
# Typed errors. The runner uses these to decide retry vs skip vs abort.
# ---------------------------------------------------------------------------
class AdapterError(Exception):
    """Base for all adapter failures."""


class TimeoutError_(AdapterError):
    """Network or generation timeout. Recoverable — retry."""


class AuthError(AdapterError):
    """Bad credentials / token. Fatal — abort the run."""


class RateLimitError(AdapterError):
    """HTTP 429 or HF rate limit. Recoverable — backoff and retry."""


class ServerError(AdapterError):
    """5xx or otherwise transient. Recoverable up to N retries."""


class GenerationError(AdapterError):
    """Model produced no output, OOM'd, etc. Treat as fatal for local mode."""


# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """Output from a single model call."""

    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # Throughput timing fields (populated when the server returns them, e.g. llama.cpp)
    prompt_n: Optional[int] = None
    prompt_ms: Optional[float] = None
    prompt_per_token_ms: Optional[float] = None
    prompt_per_second: Optional[float] = None
    predicted_n: Optional[int] = None
    predicted_ms: Optional[float] = None
    predicted_per_token_ms: Optional[float] = None
    predicted_per_second: Optional[float] = None


# ---------------------------------------------------------------------------
class ModelAdapter(ABC):
    """Minimal contract."""

    name: str = "base"

    @abstractmethod
    def generate(self, messages: list[ChatMessage], **gen_kwargs) -> GenerationResult:
        """Return the model output and token usage."""
        raise NotImplementedError

    def close(self) -> None:
        """Release sessions, GPU memory, etc. Default is a no-op."""
        return None
