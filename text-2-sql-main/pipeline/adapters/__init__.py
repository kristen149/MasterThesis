"""Adapter factory — single place that decides which adapter to build."""
from __future__ import annotations

from ..config import RunConfig
from .base import (
    AdapterError,
    AuthError,
    GenerationError,
    GenerationResult,
    ModelAdapter,
    RateLimitError,
    ServerError,
    TimeoutError_,
)
from .hf import HFAdapter
from .local_api import LocalAPIAdapter


def build_adapter(cfg: RunConfig) -> ModelAdapter:
    if cfg.source == "local":
        return LocalAPIAdapter(
            api_url=cfg.api_url,
            model_name=cfg.model,
            username=cfg.username,
            password=cfg.password,
            timeout=cfg.request_timeout,
            retries=cfg.request_retries,
        )
    if cfg.source == "hf":
        return HFAdapter(
            model_name=cfg.model,
            hf_token=cfg.hf_token,
            mode=cfg.hf_mode,
            timeout=cfg.request_timeout,
        )
    raise ValueError(f"Unknown source '{cfg.source}'. Use 'local' or 'hf'.")


__all__ = [
    "ModelAdapter",
    "GenerationResult",
    "LocalAPIAdapter",
    "HFAdapter",
    "AdapterError",
    "AuthError",
    "GenerationError",
    "RateLimitError",
    "ServerError",
    "TimeoutError_",
    "build_adapter",
]
