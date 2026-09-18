"""Recovery-backend registry — maps a config name to a RecoveryBackend factory.

Mirrors ``nlss.llm.registry``: benchmark/loop code depends only on
``RecoveryBackend`` and selects a concrete backend through this registry (never
by importing a backend module directly).  The Phase 2 bake-off froze ``rbf_gp``
as the reference backend; the registry keeps that selection a config concern.
"""

from __future__ import annotations

from typing import Callable

from .base import RecoveryBackend

BACKEND_FACTORIES: dict[str, Callable[..., RecoveryBackend]] = {}


def register_backend(name: str, factory: Callable[..., RecoveryBackend]) -> None:
    BACKEND_FACTORIES[name] = factory


def create_backend(name: str, **kwargs) -> RecoveryBackend:
    if name not in BACKEND_FACTORIES:
        raise KeyError(f"unknown recovery backend: {name}")
    return BACKEND_FACTORIES[name](**kwargs)


def _register_builtins() -> None:
    from .graph_ensemble import GraphEnsembleBackend
    from .graph_matern import GraphMaternBackend
    from .laplacian import LaplacianBackend
    from .mean import MeanBackend
    from .multiscale_diffusion import MultiScaleDiffusionBackend
    from .rbf_gp import RBFGPBackend

    register_backend("mean", MeanBackend)
    register_backend("laplacian", LaplacianBackend)
    register_backend("rbf_gp", RBFGPBackend)
    register_backend("graph_matern", GraphMaternBackend)
    register_backend("multiscale_diffusion", MultiScaleDiffusionBackend)
    register_backend("graph_ensemble", GraphEnsembleBackend)


_register_builtins()
