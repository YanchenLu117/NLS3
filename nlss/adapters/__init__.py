"""NLSS adapters.

Exposes the adapter registry (P0-2): ``register_adapter`` / ``get_adapter`` /
``available_adapters`` / ``create_adapter_from_config``.  The concrete Hypospace
and MADE adapters remain importable through their own subpackages.
"""

from .registry import (
    available_adapters,
    create_adapter_from_config,
    get_adapter,
    register_adapter,
)

__all__ = [
    "register_adapter",
    "get_adapter",
    "available_adapters",
    "create_adapter_from_config",
]
