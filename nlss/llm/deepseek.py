"""DeepSeek V4 Flash provider (OpenAI-compatible chat-completions endpoint).

Credentials are process-only.  The current deployment routes DeepSeek V4 Flash
through an OpenAI-compatible gateway; ``base_url`` and ``proxy_url`` are
configurable so the provider stays vendor-neutral at the call site.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import urllib.error
import urllib.request
from typing import Any

from .base import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


class LLMProviderError(RuntimeError):
    pass


class DeepSeekProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str | None = None,
        *,
        proxy_url: str | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url or os.environ.get("NLSS_LLM_BASE_URL", "")
        self._proxy_url = proxy_url
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await asyncio.to_thread(self._generate_sync, request)

    # -- sync core -----------------------------------------------------------
    def _generate_sync(self, request: LLMRequest) -> LLMResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if request.response_schema is not None:
            # The gateway requires the word "json" in the prompt for
            # response_format=json_object.  Inject a minimal, vendor-neutral hint.
            messages[-1]["content"] = messages[-1]["content"] + "\n\nRespond with valid JSON."
            payload_response_format = {"type": "json_object"}
        else:
            payload_response_format = None

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if payload_response_format is not None:
            payload["response_format"] = payload_response_format
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.metadata.get("thinking_disabled"):
            # Engineering opt-in: run the reasoning model without the
            # reasoning pass (faster, deterministic structured output).
            payload["thinking"] = {"type": "disabled"}
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.json_schema,
                    },
                }
                for t in request.tools
            ]

        handlers = []
        if self._proxy_url:
            handlers.append(
                urllib.request.ProxyHandler(
                    {"http": self._proxy_url, "https": self._proxy_url}
                )
            )
        opener = urllib.request.build_opener(*handlers)

        retryable = {408, 409, 429, 500, 502, 503, 504}
        last_error = ""
        body_bytes = json.dumps(payload).encode("utf-8")
        for attempt in range(self._max_attempts):
            req = urllib.request.Request(
                self._base_url,
                data=body_bytes,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "NLSS-V6/0.1",
                },
            )
            try:
                with opener.open(req, timeout=self._timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return self._parse(body)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in retryable:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < self._max_attempts:
                time.sleep(0.5 * 2**attempt + random.random() * 0.1)
        raise LLMProviderError(last_error or "LLM request failed")

    def _parse(self, body: dict[str, Any]) -> LLMResponse:
        choices = body.get("choices", [])
        if not choices:
            raise LLMProviderError(f"no choices in response: {str(body)[:300]}")
        message = choices[0].get("message", {})
        text = message.get("content") or ""
        usage_raw = body.get("usage", {}) or {}
        usage = LLMUsage(
            input_tokens=usage_raw.get("prompt_tokens"),
            output_tokens=usage_raw.get("completion_tokens"),
        )
        parsed = None
        if text.strip():
            stripped = text.strip()
            if stripped.startswith("```"):
                lines = stripped.splitlines()
                stripped = "\n".join(lines[1:-1]).strip()
                if stripped.lower().startswith("json"):
                    stripped = stripped[4:].lstrip()
            try:
                candidate = json.loads(stripped)
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                parsed = None
        tool_calls = tuple(message.get("tool_calls") or ())
        return LLMResponse(
            text=text,
            parsed=parsed,
            tool_calls=tool_calls,
            finish_reason=choices[0].get("finish_reason"),
            usage=usage,
            provider=self.provider_name,
            model=str(body.get("model", self._model)),
            raw_response_id=body.get("id"),
        )


# ---------------------------------------------------------------------------
# V7 — DeepseekV4Flash (LLMBackend) over the OpenAI-compatible vLLM gateway
# ---------------------------------------------------------------------------

# (appended to the V6 DeepSeekProvider module; original code above unchanged)

import os

from typing import Mapping

from .base import GenerationResult, LLMBackend


def _default_api_key() -> str:
    """Resolve the DeepSeek API key from the process environment.

    Never a literal in source.  Order: NLSS_LLM_API_KEY, then DEEPSEEK_API_KEY.
    """
    for name in ("NLSS_LLM_API_KEY", "DEEPSEEK_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise ValueError(
        "DeepseekV4Flash requires an API key: set NLSS_LLM_API_KEY or "
        "DEEPSEEK_API_KEY (see configs/llm/ and llm_env.sh)."
    )


# Default gateway for the deployed vLLM endpoint (FROZEN, per architecture).
DEFAULT_V4FLASH_BASE_URL = os.environ.get("NLSS_LLM_BASE_URL", "")
DEFAULT_V4FLASH_MODEL = "deepseek-v4-flash"
DEFAULT_V4FLASH_CONTEXT = 512 * 1024  # 512K context window
DEFAULT_V4FLASH_MAX_OUTPUT = 8192  # default cap; caller may raise to 32768


class DeepseekV4Flash(LLMBackend):
    """Extensible default LLMBackend — DeepSeek V4 Flash via OpenAI-compatible
    vLLM gateway, using the openai SDK (present in the nlss-core env).

    Configuration (see also configs/llm/deepseek_v4_flash.yaml):

        base_url          : default from NLSS_LLM_BASE_URL env
        api_key           : process-only, from env (never stored here)
        model             : deepseek-v4-flash
        context_window    : 512K
        max_output_tokens : 8192 (cap range 8192..32768)

    Extensibility: a new model (GPT/Claude/…) is one new ``LLMBackend``
    subclass + one `register_backend` line; core is untouched.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_V4FLASH_MODEL,
        base_url: str | None = None,
        *,
        max_output_tokens: int = DEFAULT_V4FLASH_MAX_OUTPUT,
        context_window: int = DEFAULT_V4FLASH_CONTEXT,
        timeout_seconds: float = 120.0,
        input_price_per_mtok: float | None = None,
        output_price_per_mtok: float | None = None,
    ) -> None:
        if not isinstance(max_output_tokens, int) or not (
            8192 <= max_output_tokens <= 32768
        ):
            raise ValueError("max_output_tokens must be within [8192, 32768]")
        resolved_key = api_key or _default_api_key()
        self._model = model
        self._base_url = (
            base_url
            or os.environ.get("NLSS_LLM_BASE_URL")
            or DEFAULT_V4FLASH_BASE_URL
        )
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout_seconds
        # Per-1M-token USD pricing; None => cost not estimated.
        self._in_price = input_price_per_mtok
        self._out_price = output_price_per_mtok
        try:
            import openai  # lazy: openai SDK is an env concern, not a hard dep
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "DeepseekV4Flash needs the openai package (see nlss-core env)"
            ) from exc
        # P1 fix: use the RESOLVED base_url (``self._base_url``), which folds in
        # the NLSS_LLM_BASE_URL env override and the frozen default.  The raw
        # ``base_url`` constructor arg is only the explicit (pre-env) value and
        # would otherwise silently bypass the env override on the client.
        self._client = openai.OpenAI(
            base_url=self._base_url,
            api_key=resolved_key,
            timeout=timeout_seconds,
        )

    # -- LLMBackend contract ------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def generate(self, system: str, user: str, **kwargs: Any) -> "GenerationResult":
        # Per-call ceiling stays within [1, min(configured_max, context_window)]
        # so a stray huge/invalid override can never reach the gateway.
        ceiling = max(1, min(int(getattr(self, "_max_output_tokens", 8192)), self._context_window))
        max_tokens = int(kwargs.get("max_output_tokens", self._max_output_tokens))
        max_tokens = min(max(max_tokens, 1), ceiling)
        temperature = kwargs.get("temperature", 0.6)
        seed = kwargs.get("seed")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": max_tokens,
        }
        if seed is not None:
            params["seed"] = seed

        resp = self._client.chat.completions.create(**params)
        message = resp.choices[0].message if getattr(resp, "choices", None) else None
        text = (message.content or "") if message else ""
        usage = getattr(resp, "usage", None)
        inp = int(getattr(usage, "prompt_tokens", 0) or 0)
        out = int(getattr(usage, "completion_tokens", 0) or 0)
        total = inp + out
        cost = self._estimate_cost(inp, out)
        return GenerationResult(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            cost=cost,
            model=self._model,
            finish_reason=getattr(resp.choices[0], "finish_reason", None)
            if getattr(resp, "choices", None)
            else None,
            raw={
                "base_url": self._base_url,
                "context_window": self._context_window,
                "id": getattr(resp, "id", None),
            },
        )

    # -- helpers ---------------------------------------------------------------

    def _estimate_cost(self, in_tokens: int, out_tokens: int) -> float | None:
        if self._in_price is None or self._out_price is None:
            return None
        return (in_tokens / 1e6) * self._in_price + (out_tokens / 1e6) * self._out_price

    # -- introspection ----------------------------------------------------------

    @property
    def context_window(self) -> int:
        return self._context_window

    def describe(self) -> Mapping[str, Any]:
        """Stable, non-secret description for config/logs."""
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "base_url": self._base_url,
            "context_window": self._context_window,
            "max_output_tokens": self._max_output_tokens,
        }
