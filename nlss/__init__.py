"""NLSS V7 — top-level three-operation facade.

Exposes the FROZEN §3.1 contract directly at ``import nlss``:

    nlss.update(evidence_batch)
    space_state = nlss.recover()
    readout_text = nlss.readout(space_state, token_budget=...)

The module holds *one* shared :class:`~nlss.core.model.NLSSModel` instance.
Inject it with :func:`configure_model` (the benchmark wiring constructs the
adapter/backend/readout/llm); the thin wrappers below then pass through to it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .core.model import EvidenceRecord, NLSSModel
from .core.representation import RepresentationParadigm, RepresentationSpecification

__all__ = [
    "NLSSModel",
    "configure_model",
    "get_model",
    "update",
    "recover",
    "readout",
    "RepresentationParadigm",
    "RepresentationSpecification",
    "__version__",
]

__version__ = "7.0.0"

_model: NLSSModel | None = None


def configure_model(model: NLSSModel) -> NLSSModel:
    """Set the shared module-level NLSSModel used by the thin wrappers."""
    global _model
    _model = model
    return model


def get_model() -> NLSSModel:
    """Return the configured shared model (raises if none configured)."""
    if _model is None:
        raise RuntimeError(
            "no NLSSModel configured; call nlss.configure_model(model) first"
        )
    return _model


# -- three-operation pass-throughs -----------------------------------------


def update(evidence_batch: Sequence[EvidenceRecord]) -> dict[str, Any]:
    """nlss.update(evidence_batch) -> snapshot (thin pass-through)."""
    return get_model().update(evidence_batch)


def recover(features: Mapping[str, Any] | None = None):
    """nlss.recover() -> space_state (thin pass-through)."""
    return get_model().recover(features)


def readout(space_state, token_budget: int = 1800) -> str:
    """nlss.readout(space_state, token_budget=...) -> str (thin pass-through)."""
    return get_model().readout(space_state, token_budget)
