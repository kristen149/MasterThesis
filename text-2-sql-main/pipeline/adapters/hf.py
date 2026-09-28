"""HuggingFace adapter.

Two modes:

  - ``api`` (default): uses ``huggingface_hub.InferenceClient.chat_completion``.
    Lightweight: no model download, no GPU required. Auth via ``hf_token``.

  - ``local``: loads the model with ``transformers.AutoModelForCausalLM``.
    Heavier (download + GPU/CPU memory), but offline-capable and gives full
    control over generation params. Requires the ``transformers`` and
    ``torch`` extras.

Both paths normalize to a single ``generate(messages) -> str`` so the runner
doesn't care which is in use.
"""
from __future__ import annotations

from typing import Any, Optional

from ..prompts.base import ChatMessage
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


class HFAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        mode: str = "api",
        device: str = "auto",
        dtype: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.name = model_name
        self.mode = mode
        self.hf_token = hf_token
        self.timeout = timeout

        if mode == "api":
            self._init_api()
        elif mode == "local":
            self._init_local(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown hf_mode '{mode}'. Use 'api' or 'local'.")

    # ----------------------------------------------------------------------
    def _init_api(self) -> None:
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except ImportError as e:
            raise AdapterError(
                "huggingface_hub is required for hf_mode=api. "
                "Install with: pip install huggingface_hub"
            ) from e

        # InferenceClient resolves auth from the explicit token if given,
        # otherwise from the env (HF_TOKEN/HUGGINGFACE_TOKEN). We pass the
        # explicit token when present so user-CLI tokens take priority.
        self._client = InferenceClient(
            model=self.name,
            token=self.hf_token,
            timeout=self.timeout,
        )

    def _init_local(self, device: str, dtype: Optional[str]) -> None:
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as e:
            raise AdapterError(
                "transformers and torch are required for hf_mode=local. "
                "Install with: pip install transformers torch"
            ) from e

        torch_dtype: Any
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(self.name, token=self.hf_token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.name,
            token=self.hf_token,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self._model.eval()

    # ----------------------------------------------------------------------
    def generate(self, messages: list[ChatMessage], **gen_kwargs) -> GenerationResult:
        if self.mode == "api":
            return self._generate_api(messages, **gen_kwargs)
        return self._generate_local(messages, **gen_kwargs)

    def _generate_api(self, messages: list[ChatMessage], **gen_kwargs) -> GenerationResult:
        try:
            response = self._client.chat_completion(
                messages=[m.to_dict() for m in messages],
                max_tokens=gen_kwargs.get("max_tokens", 256),
                temperature=gen_kwargs.get("temperature", 0.0),
                top_p=gen_kwargs.get("top_p", 1.0),
            )
        except Exception as e:
            # huggingface_hub raises different exception classes across versions;
            # we sniff the message to map to our typed errors.
            msg = str(e).lower()
            if "timeout" in msg or "timed out" in msg:
                raise TimeoutError_(str(e)) from e
            if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
                raise AuthError(str(e)) from e
            if "429" in msg or "rate" in msg:
                raise RateLimitError(str(e)) from e
            if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
                raise ServerError(str(e)) from e
            raise GenerationError(str(e)) from e

        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as e:
            raise GenerationError(f"Unexpected HF response shape: {response!r}") from e

        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    def _generate_local(self, messages: list[ChatMessage], **gen_kwargs) -> GenerationResult:
        try:
            import time
            import torch  # type: ignore
        except ImportError as e:
            raise AdapterError("torch missing") from e

        # Most modern chat models ship a chat template — apply it for fidelity.
        try:
            prompt = self._tokenizer.apply_chat_template(
                [m.to_dict() for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: naive concat. Better-than-nothing for models without templates.
            prompt = "\n".join(f"[{m.role}] {m.content}" for m in messages) + "\n[assistant]"

        t0 = time.perf_counter()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        try:
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get("max_tokens", 256),
                    temperature=max(gen_kwargs.get("temperature", 0.0), 1e-5),
                    do_sample=gen_kwargs.get("temperature", 0.0) > 0,
                    top_p=gen_kwargs.get("top_p", 1.0),
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            # OOM, CUDA errors, etc. Fatal — every subsequent sample will repeat.
            raise GenerationError(f"Generation crashed: {e}") from e
        predicted_ms = (time.perf_counter() - t1) * 1000.0

        # Strip the prompt prefix so we return only the model's continuation.
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = out[0][prompt_len:]
        completion_tokens = generated_ids.shape[0]
        total_tokens = prompt_len + completion_tokens
        return GenerationResult(
            text=self._tokenizer.decode(generated_ids, skip_special_tokens=True),
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_n=prompt_len,
            prompt_ms=round(prompt_ms, 3),
            prompt_per_token_ms=round(prompt_ms / prompt_len, 6) if prompt_len else None,
            prompt_per_second=round(prompt_len / (prompt_ms / 1000), 2) if prompt_ms else None,
            predicted_n=completion_tokens,
            predicted_ms=round(predicted_ms, 3),
            predicted_per_token_ms=round(predicted_ms / completion_tokens, 6) if completion_tokens else None,
            predicted_per_second=round(completion_tokens / (predicted_ms / 1000), 2) if predicted_ms else None,
        )

    # ----------------------------------------------------------------------
    def close(self) -> None:
        if self.mode == "local":
            try:
                import torch  # type: ignore
                del self._model
                del self._tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
