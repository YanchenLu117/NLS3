"""GB1 benchmark adapter (§10).

Exposes :class:`GB1Adapter` (with configurable solution criterion: top-5%
``top_pct`` primary, or §10.4 better-than-WT ``better_wt``) and, via the
shared adapter registry in :mod:`nlss.adapters.registry`, registers it under
``gb1`` so ``get_adapter("gb1")`` / ``create_adapter_from_config("gb1", cfg)``
work.

Registration authority: only ``registry._register_builtins`` registers the GB1
factory (the single authoritative registration point, 委员C P1 merge); this
module only *defines* ``_gb1_factory`` for use there.  Previously this module
defined AND eagerly registered its own ``_gb1_factory`` that silently DROPPED
``criterion`` (``return GB1Adapter(q=q)``) and called ``register()`` at import
time — clobbering the registry's criterion-forwarding factory on the lazy
import that ``GB1Adapter`` triggers: the 1st
``create_adapter_from_config("gb1", {"criterion": "better_wt"})`` returned
better_wt, the 2nd returned top_pct.  That P0 registration clobber is fixed
here: this module no longer self-registers, matching the BH adapter pattern so
``create_adapter_from_config`` returns a stable, criterion-correct arm.
"""

from __future__ import annotations

from typing import Any, Mapping

from .adapter import GB1Adapter
from .data import GB1Data, WILD_TYPE

__all__ = ["GB1Adapter", "GB1Data", "WILD_TYPE", "_gb1_factory"]


def _gb1_factory(cfg: Mapping[str, Any] | None) -> GB1Adapter:
    """Construct GB1Adapter from a config mapping (used by the registry).

    Accepts the config-driven solution-criterion keys: ``q`` (top-``(1-q)``
    ``top_pct`` primary, default 0.95) and ``criterion`` (``top_pct`` |
    ``better_wt``).  Forwarding ``criterion`` through the factory is what makes
    ``create_adapter_from_config("gb1", {"criterion": "better_wt"})`` stable.
    """
    cfg = dict(cfg or {})
    kwargs: dict[str, Any] = {"q": float(cfg.get("q", 0.95))}
    if "criterion" in cfg:
        kwargs["criterion"] = str(cfg["criterion"])
    return GB1Adapter(**kwargs)
