"""NLSS LLM layer (V6 LLMProvider + V7 LLMBackend).

V7 exposes:
    - ``BaseLLM``/``LLMBackend`` (extensible contract): ``generate(system, user)``
    - ``GenerationResult`` (text + C_in/C_out/C_total + estimated cost)
    - model-name-backed registry: ``register_backend`` / ``get_backend`` /
      ``create_backend_from_config``
"""

from .base import (
    GenerationResult,
    LLMBackend,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)
from .registry import (
    available_backends,
    create_backend_from_config,
    create_provider,
    get_backend,
    register_backend,
    register_provider,
)

# Convenience alias: the V7 extensible contract is often named ``BaseLLM``.
BaseLLM = LLMBackend

__all__ = [
    # contract
    "LLMBackend",
    "BaseLLM",
    "GenerationResult",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "LLMMessage",
    # registry
    "register_backend",
    "get_backend",
    "available_backends",
    "create_backend_from_config",
    "register_provider",
    "create_provider",
]
