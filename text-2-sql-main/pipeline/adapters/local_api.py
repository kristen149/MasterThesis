"""Local OpenAI-compatible API adapter.

Targets servers that expose the OpenAI ``/v1/chat/completions`` shape — the
local Qwen deployment in the spec is the primary use case, but the adapter
also works with vLLM, llama.cpp's OpenAI server, LM Studio, or anything
following the same JSON contract.

Auth
----
Uses HTTP Basic auth with (username, password) when both are provided.
Credentials may come from the CLI or from ``T2S_API_USERNAME`` /
``T2S_API_PASSWORD`` environment variables (resolved in ``RunConfig``).

Retries
-------
- 429 / 5xx → urllib3 ``Retry`` with exponential backoff.
- Connection / read timeouts → raised as ``TimeoutError_`` for the runner.
- 401 / 403 → raised as ``AuthError`` and not retried.
"""
from __future__ import annotations

from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from ..prompts.base import ChatMessage
from .base import (
    AuthError,
    GenerationError,
    GenerationResult,
    ModelAdapter,
    RateLimitError,
    ServerError,
    TimeoutError_,
)


class LocalAPIAdapter(ModelAdapter):
    def __init__(
        self,
        api_url: str,
        model_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 60,
        retries: int = 3,
    ) -> None:
        self.api_url = api_url
        self.name = model_name
        self.timeout = timeout

        self.auth = (
            HTTPBasicAuth(username, password)
            if (username and password)
            else None
        )

        self.session = requests.Session()
        # Auto-retry only on 429 / 5xx; auth and client errors fail fast.
        retry = Retry(
            total=retries,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("POST",),
            raise_on_status=False,
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    # ----------------------------------------------------------------------
    def generate(self, messages: list[ChatMessage], **gen_kwargs) -> GenerationResult:
        # Always stop on Qwen's end-of-turn token so the model doesn't leak
        # ChatML tokens or hallucinate extra conversation turns.
        default_stop = ["<|im_end|>", "<|endoftext|>"]
        caller_stop = gen_kwargs.get("stop")
        if caller_stop:
            # Merge caller-supplied stops with the defaults, deduplicating.
            stop = list(dict.fromkeys(default_stop + list(caller_stop)))
        else:
            stop = default_stop

        payload = {
            "model": self.name,
            "messages": [m.to_dict() for m in messages],
            "temperature": gen_kwargs.get("temperature", 0.0),
            "max_tokens": gen_kwargs.get("max_tokens", 256),
            "stop": stop,
        }
        # Forward extra kwargs (top_p, presence_penalty, frequency_penalty, seed) if provided.
        for k in ("top_p", "presence_penalty", "frequency_penalty", "seed"):
            if k in gen_kwargs and gen_kwargs[k] is not None:
                payload[k] = gen_kwargs[k]

        try:
            resp = self.session.post(
                self.api_url,
                json=payload,
                auth=self.auth,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            raise TimeoutError_(f"Request to {self.api_url} timed out") from e
        except requests.exceptions.ConnectionError as e:
            raise ServerError(f"Connection error to {self.api_url}: {e}") from e

        status = resp.status_code
        if status in (401, 403):
            raise AuthError(f"Authentication failed ({status}) for {self.api_url}")
        if status == 429:
            raise RateLimitError(f"Rate-limited by {self.api_url}")
        if 500 <= status < 600:
            raise ServerError(f"Server error {status}: {resp.text[:300]}")
        if status >= 400:
            raise GenerationError(f"HTTP {status}: {resp.text[:300]}")

        try:
            data = resp.json()
        except ValueError as e:
            raise GenerationError(f"Non-JSON response: {resp.text[:300]}") from e

        try:
            text = data["choices"][0]["message"]["content"]

        except (KeyError, IndexError, TypeError) as e:
            raise GenerationError(f"Unexpected response shape: {data}") from e

        usage = data.get("usage") or {}
        t = data.get("timings") or {}
        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            prompt_n=t.get("prompt_n"),
            prompt_ms=t.get("prompt_ms"),
            prompt_per_token_ms=t.get("prompt_per_token_ms"),
            prompt_per_second=t.get("prompt_per_second"),
            predicted_n=t.get("predicted_n"),
            predicted_ms=t.get("predicted_ms"),
            predicted_per_token_ms=t.get("predicted_per_token_ms"),
            predicted_per_second=t.get("predicted_per_second"),
        )

    def close(self) -> None:
        self.session.close()
