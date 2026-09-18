"""LLM registry — maps a MODEL name to an :class:`LLMBackend` factory (V7),
and keeps the V6 provider registry for backward compatibility.

Existing benchmark/loop code continues to use :class:`LLMProvider` through the
``provider`` key.  The V7 ``backend`` registry is the extensibility seam: adding
GPT / Claude / any OpenAI-compatible model = one new ``LLMBackend`` subclass +
one ``register_backend`` line, resolved by model name from config.  Core never
imports a vendor module directly.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping

from .base import LLMBackend, LLMProvider

# ---------------------------------------------------------------------------
# V7 — backend registry (by model name)
# ---------------------------------------------------------------------------

BACKEND_FACTORIES: dict[str, Callable[..., LLMBackend]] = {}


def register_backend(name: str, factory: Callable[..., LLMBackend]) -> None:
    """Register an :class:`LLMBackend` factory under a model name."""
    BACKEND_FACTORIES[name] = factory


def get_backend(name: str, **kwargs) -> LLMBackend:
    """Instantiate an :class:`LLMBackend` by model name."""
    if name not in BACKEND_FACTORIES:
        raise KeyError(f"unknown LLM backend: {name}")
    return BACKEND_FACTORIES[name](**kwargs)


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(BACKEND_FACTORIES))


# Config-driven instantiation from a yaml file (model name -> backend).
def create_backend_from_config(path: str, **overrides) -> LLMBackend:
    """Load a yaml config and build the backend named by its ``model`` key.

    The optional ``api_key`` is ignored here (resolved from the env inside the
    backend factory); never store secrets in the yaml.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    model = str(cfg.get("model", "deepseek-v4-flash"))
    # Top-level backend kwargs (base_url / max_output_tokens / …) plus an
    # optional ``params`` sub-map; never forward secrets (api_key/api_key).
    params: dict[str, Any] = dict(cfg.get("params", {}) or {})
    for key, value in cfg.items():
        if key in ("model", "provider", "api_key", "params"):
            continue
        params.setdefault(key, value)
    params.update(overrides)
    return get_backend(model, **params)


# ---------------------------------------------------------------------------
# V6 — provider registry (frozen, unchanged semantics)
# ---------------------------------------------------------------------------


PROVIDER_FACTORIES: dict[str, Callable[..., LLMProvider]] = {}


def register_provider(name: str, factory: Callable[..., LLMProvider]) -> None:
    PROVIDER_FACTORIES[name] = factory


def create_provider(name: str, **kwargs) -> LLMProvider:
    if name not in PROVIDER_FACTORIES:
        raise KeyError(f"unknown LLM provider: {name}")
    return PROVIDER_FACTORIES[name](**kwargs)


def _register_builtins() -> None:
    from .deepseek import DeepSeekProvider, DeepseekV4Flash

    register_provider("deepseek", DeepSeekProvider)
    # V7: default model name -> DeepseekV4Flash backend.
    register_backend("deepseek-v4-flash", DeepseekV4Flash)


_register_builtins()
