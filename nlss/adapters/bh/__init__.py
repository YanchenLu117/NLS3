"""BH benchmark adapter (Buchwald-Hartwig HTE, §9).

Exposes :class:`BHAdapter` (with configurable solution arm: top-5% primary,
§9.4 top-10% secondary, or fixed-yield cut) and, via the shared adapter
registry in :mod:`nlss.adapters.registry`, registers it under ``bh`` so
``get_adapter("bh")`` / ``create_adapter_from_config("bh", cfg)`` work.

Registration authority: only ``registry._register_builtins`` registers the BH
factory (the single authoritative registration point, 委员C P1 merge); this
module only *defines* the ``_bh_factory`` used there.
"""

from __future__ import annotations

from typing import Any, Mapping

from .adapter import BHAdapter
from .data import BHData, Candidate

__all__ = ["BHAdapter", "BHData", "Candidate"]


def _bh_factory(cfg: Mapping[str, Any] | None) -> BHAdapter:
    """Construct BHAdapter from a config mapping (used by the registry).

    Accepts the config-driven solution arm keys: ``q`` (top-``(1-q)`` primary),
    ``criterion`` (``top_pct`` | ``fixed``), ``top_frac``, ``fixed_yield``.
    """
    cfg = dict(cfg or {})
    kwargs: dict[str, Any] = {"q": float(cfg.get("q", 0.95))}
    if "criterion" in cfg:
        kwargs["criterion"] = str(cfg["criterion"])
    if cfg.get("top_frac") is not None:
        kwargs["top_frac"] = float(cfg["top_frac"])
    if cfg.get("fixed_yield") is not None:
        kwargs["fixed_yield"] = float(cfg["fixed_yield"])
    return BHAdapter(**kwargs)
