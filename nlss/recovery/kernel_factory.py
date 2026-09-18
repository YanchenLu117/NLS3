"""V7 Core — kernel-per-representation factory (EXPERIMENT_PLAN §2.1 Layer II).

The plan requires the empirical field's kernel to be DETERMINED BY THE ADMITTED
representation's geometry (K), not a fixed default: Euclidean → RBF/Matern,
sequence → Hamming/string, graph → graph kernel, mixed typed state → per-field
composite.  ``resolve_kernel`` maps an admitted :class:`RepresentationSpecification`
(its ``geometry`` and ``backend_hint``) to a recovery backend, defaulting to
``laplacian`` as the documented degraded/empty fallback.  Provenance is recorded
so every choice is auditable (Attack H / §12.2 field provenance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core.representation import RepresentationSpecification


@dataclass(frozen=True, slots=True)
class KernelDecision:
    """The resolved empirical-field substrate for an admitted contract."""

    backend: str
    kernel: str
    features_scheme: str  # 'euclidean' | 'graph' | 'sequence' | 'mixed' | 'none'
    geometry_kind: str  # from spec.geometry
    provenance: Mapping[str, Any] = field(default_factory=dict)


def resolve_kernel(
    spec: "RepresentationSpecification | None",
    *,
    override: str | None = None,
) -> KernelDecision:
    """Map a representation's geometry to a recovery backend + kernel.

    - ``override`` forces a backend regardless of geometry (e.g. an arm's fixed
      representation-paradigm substrate for the V1/V6 ablation).
    - Geometry is read from ``spec.geometry`` dict (kernel/neighborhood/features)
      and ``spec.backend_hint``.
    - Falls back to ``laplacian`` (graph) whenever the geometry is unset or
      unknown — the documented degraded/empty fallback, never a silent guess.
    """
    if spec is None:
        return KernelDecision(
            backend="laplacian", kernel="graph-laplacian", features_scheme="graph",
            geometry_kind="none", provenance={"reason": "no contract -> fallback"},
        )

    geometry = spec.geometry or {}
    geom_kernel = str(geometry.get("kernel", "")).lower() if geometry else ""
    hint = spec.backend_hint

    if override is not None:
        backend = override
    elif "rbf" in geom_kernel or "matern" in geom_kernel:
        backend = "rbf_gp"
        if hint:
            backend = hint  # explicit backend_hint governs when set
    elif hint in ("rbf_gp", "graph_matern", "multiscale_diffusion", "graph_ensemble", "mean"):
        backend = hint
    else:
        backend = "laplacian"

    # features_scheme derived from the chosen backend + geometry kind
    kind = "none" if not geometry else str(geometry.get("kernel", "graph")).lower()
    if backend == "rbf_gp":
        scheme = "euclidean"
    elif backend == "mean":
        scheme = "none"
    else:
        scheme = "graph"
    if "ham" in kind or "string" in kind or "seq" in kind:
        scheme = "sequence"
    if "mixed" in kind:
        scheme = "mixed"

    return KernelDecision(
        backend=backend,
        kernel=geom_kernel or ("euclidean" if scheme == "euclidean" else "graph-laplacian"),
        features_scheme=scheme,
        geometry_kind=kind,
    provenance={
            "geometry_used": dict(geometry) if geometry else None,
            "backend_hint": hint,
            "override": override,
            "reason": ("explicit_override" if override else
                       "geometry_kernel" if ("rbf" in geom_kernel or "matern" in geom_kernel) else
                       ("explicit_hint" if hint else "fallback_laplacian")),
        },
    )
