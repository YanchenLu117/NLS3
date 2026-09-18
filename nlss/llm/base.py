"""Provider-neutral LLM interface (V6 + V7).

V6 (unchanged): the ``LLMProvider`` abstraction used by the frozen TaskAdapter
``compile`` path — vendor-neutral ``LLMRequest`` / ``LLMResponse`` dataclasses.

V7 (additive): the extensible ``LLMBackend`` contract and ``GenerationResult``
for the three-operation facade:

    backend.generate(system, user, **kwargs) -> GenerationResult

``GenerationResult`` carries the text plus token metering (C_in / C_out /
C_total, aligned with the technical report §4.3) and an estimated dollar cost.
Every vendor (DeepSeek, GPT, Claude, …) plugs in as a new ``LLMBackend``
subclass registered by model name in ``nlss.llm.registry``; core never imports
a vendor module directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# V6 — LLMProvider (frozen, unchanged)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    json_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    temperature: float = 0.6
    max_output_tokens: int = 4096
    tools: tuple[ToolSpec, ...] = ()
    response_schema: Mapping[str, Any] | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None = None
    estimated_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    parsed: Mapping[str, Any] | None
    tool_calls: tuple[Mapping[str, Any], ...]
    finish_reason: str | None
    usage: LLMUsage
    provider: str
    model: str
    raw_response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    async def generate_json(self, request: LLMRequest | None = None, **kwargs) -> LLMResponse:
        """Generate and parse JSON.

        Canonical form: ``generate_json(request)`` with ``request.response_schema``
        set.  A backward-compatible convenience is also accepted for adapters that
        call ``generate_json(prompt=..., response_schema=..., ...)`` (or provide
        ``messages=...`` directly): it is wrapped into an :class:`LLMRequest` on
        the caller's behalf.
        """
        if request is None:
            prompt = kwargs.pop("prompt", None)
            messages = tuple(kwargs.pop("messages", ()))
            if prompt is None and not messages:
                raise ValueError(
                    "generate_json requires a request, or a 'prompt'/'messages' kwarg"
                )
            if not messages:
                messages = (LLMMessage(role="user", content=str(prompt)),)
            request = LLMRequest(
                messages=messages,
                temperature=kwargs.pop("temperature", 0.6),
                max_output_tokens=kwargs.pop("max_output_tokens", 4096),
                response_schema=kwargs.pop("response_schema", None),
                seed=kwargs.pop("seed", None),
                metadata=kwargs,
            )
        if request.response_schema is None:
            raise ValueError("generate_json requires response_schema.")
        return await self.generate(request)


# ---------------------------------------------------------------------------
# V7 — LLMBackend (extensible contract) + GenerationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A completed generation: text plus token metering and estimated cost.

    ``input_tokens`` / ``output_tokens`` / ``total_tokens`` map to the technical
    report §4.3 C_in / C_out / C_total.  ``cost`` is an ESTIMATED dollar cost
    (USD) derived from per-model pricing when known, else ``None``.
    """

    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float | None = None

    model: str | None = None
    finish_reason: str | None = None
    raw: Mapping[str, Any] | None = field(default_factory=dict)


class LLMBackend(ABC):
    """Extensible vendor-neutral LLM contract (V7).

    ``generate(system, user, **kwargs)`` returns a :class:`GenerationResult`.
    ``kwargs`` may carry ``temperature`` / ``max_output_tokens`` / ``seed`` etc.
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def generate(self, system: str, user: str, **kwargs: Any) -> GenerationResult: ...
