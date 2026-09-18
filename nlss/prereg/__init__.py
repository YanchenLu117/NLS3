"""V8 P0 — preregistration registry (Detail §2.2).

Public surface: :class:`PreregRegistry`, :data:`SCHEMA_VERSION`,
:meth:`load_registry`, :meth:`verify_registry_hash`,
:meth:`canonical_registry_bytes`.
"""

from .registry import (
    SCHEMA_VERSION,
    BlockedCellError,
    FrozenRegistryError,
    PreregRegistry,
    canonical_registry_bytes,
    load_registry,
    verify_registry_hash,
)

__all__ = [
    "SCHEMA_VERSION",
    "BlockedCellError",
    "FrozenRegistryError",
    "PreregRegistry",
    "canonical_registry_bytes",
    "load_registry",
    "verify_registry_hash",
]
