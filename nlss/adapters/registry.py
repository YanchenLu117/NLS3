"""Adapter registry — maps a config name to a TaskAdapter factory (P0-2).

Mirrors ``nlss.llm.registry`` / ``nlss.recovery.registry``: benchmark/loop code
depends only on :class:`TaskAdapter` and selects a concrete adapter through this
registry rather than importing an adapter module directly.

SINGLE REGISTRATION AUTHORITY (委员C P1 merge): this module's
``_register_builtins`` is the ONE place every adapter factory is registered.
Earlier revisions had each adapter subpackage (bh/gb1) also
define its own factory + register() and call register() at import time,
producing a functionally-identical *duplicate* registration (stylistic
duplication, overwriting the same name with the same factory — never a
functional conflict, but a drift/confusion risk).  Per the P1 cleanup the
subpackage-level self-registration was removed; each __init__.py now only
defines its factory, and _register_builtins here is authoritative.
available_adapters(): hypospace, made, bh, gb1
(PiEvo stays intentionally unregistered — honest stub).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core.interfaces import TaskAdapter

ADAPTER_FACTORIES: dict[str, Callable[[Mapping[str, Any] | None], "TaskAdapter"]] = {}


def register_adapter(name: str, factory) -> None:
    """Register a ``factory(cfg: Mapping | None) -> TaskAdapter`` under a name."""
    ADAPTER_FACTORIES[name] = factory


def get_adapter(name: str):
    """Return the registered adapter factory for ``name`` (raises KeyError).

    The returned value is a callable ``factory(cfg)`` that constructs the
    :class:`TaskAdapter`; use :func:`create_adapter_from_config` to construct.
    """
    if name not in ADAPTER_FACTORIES:
        raise KeyError(f"unknown adapter: {name}")
    return ADAPTER_FACTORIES[name]


def available_adapters() -> tuple[str, ...]:
    return tuple(sorted(ADAPTER_FACTORIES))


def create_adapter_from_config(name: str, cfg: Mapping[str, Any] | None = None) -> "TaskAdapter":
    """Instantiate the :class:`TaskAdapter` named by ``name`` from ``cfg``.

    ``cfg`` supplies the adapter-construction arguments (see each registered
    factory).  ``cfg=None`` falls back to a built-in minimal default so bare
    lookup still constructs.
    """
    if name not in ADAPTER_FACTORIES:
        raise KeyError(f"unknown adapter: {name}")
    return ADAPTER_FACTORIES[name](dict(cfg) if cfg else None)


# ---------------------------------------------------------------------------
# Built-in registrations (existing adapters only — no implementation changes)
# ---------------------------------------------------------------------------


_DEFAULT_BOOLEAN_TASK = {
    "dataset": "hypospace_registry",
    "observation_set_id": "registry_default",
    "variables": ("x", "y"),
    "operators": frozenset({"AND", "OR", "NOT"}),
    "max_depth": 2,
    "mechanistic_opts": {
        "apply_commutativity": True,
        "apply_idempotence_and_or": True,
        "flatten_associativity": True,
    },
    "observations": (),
    "n_observations": 0,
}


def _hypospace_factory(cfg: Mapping[str, Any] | None) -> "TaskAdapter":
    from .hypospace.boolean import BooleanAdapter
    from .hypospace.causal import CausalAdapter
    from .hypospace.voxel3d import Voxel3DAdapter
    from .hypospace.tasks import BooleanTask, CausalTask, VoxelTask

    cfg = dict(cfg or {})
    domain = str(cfg.get("domain", "boolean"))
    task_cfg = dict(cfg.get("task") or _DEFAULT_BOOLEAN_TASK)
    if domain == "causal":
        return CausalAdapter(CausalTask(**task_cfg))
    if domain == "voxel3d":
        return Voxel3DAdapter(VoxelTask(**task_cfg))
    return BooleanAdapter(BooleanTask(**task_cfg))


def _made_factory(cfg: Mapping[str, Any] | None) -> "TaskAdapter":
    from .made import MADEAdapter

    cfg = dict(cfg or {})
    elements = tuple(cfg.get("elements", ("Mg", "Sn", "Sr")))
    params = dict(cfg.get("params") or {})
    return MADEAdapter(elements, **params)


def _bh_factory(cfg: Mapping[str, Any] | None) -> "TaskAdapter":
    # NOTE (委员A P0): BH solution arm is config-driven.  The frozen primary is
    # top-5% (q=0.95).  The §9.4 OPTIONAL secondary-significance arms are selected
    # via `criterion`:
    #   top_pct + top_frac=0.10  -> top-10% secondary arm
    #   fixed   + fixed_yield=80 -> absolute-yield secondary arm
    # Registering here (single authority, see module docstring on P1 merge) means
    # a config with these keys constructs the right arm through the registry.
    from .bh.adapter import BHAdapter

    cfg = dict(cfg or {})
    kwargs: dict[str, Any] = {"q": float(cfg.get("q", 0.95))}
    if "criterion" in cfg:
        kwargs["criterion"] = str(cfg["criterion"])
    if cfg.get("top_frac") is not None:
        kwargs["top_frac"] = float(cfg["top_frac"])
    if cfg.get("fixed_yield") is not None:
        kwargs["fixed_yield"] = float(cfg["fixed_yield"])
    return BHAdapter(**kwargs)


def _gb1_factory(cfg: Mapping[str, Any] | None) -> "TaskAdapter":
    from .gb1.adapter import GB1Adapter

    cfg = dict(cfg or {})
    kwargs: dict[str, Any] = {"q": float(cfg.get("q", 0.95))}
    if "criterion" in cfg:
        kwargs["criterion"] = str(cfg["criterion"])
    return GB1Adapter(**kwargs)


def _sciexplorer_factory(cfg):
    from .sciexplorer import SciExplorerPhysicsAdapter
    return SciExplorerPhysicsAdapter(config=cfg)


def _register_builtins() -> None:
    register_adapter("hypospace", _hypospace_factory)
    register_adapter("made", _made_factory)
    register_adapter("sciexplorer", _sciexplorer_factory)
    # T2 benchmark adapters (P0): BH, GB1.
    # PiEvo is intentionally NOT registered: its status is an honest stub that
    # requires a dedicated torch/rdkit env (see src/nlss/adapters/pievo/__init__.py).
    register_adapter("bh", _bh_factory)
    register_adapter("gb1", _gb1_factory)


_register_builtins()
