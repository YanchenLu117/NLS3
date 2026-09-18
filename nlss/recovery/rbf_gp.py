"""Euclidean / RBF GP backend (B2) — representation control baseline.

Fits a standard RBF-kernel GP directly on the scientific feature vectors
(``RecoveryInput.features``), ignoring the graph.  This isolates whether the
graph structure adds anything beyond a Euclidean kernel on the raw features.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.errors import RecoveryNotFitted
from ..core.types import PosteriorView
from .base import RecoveryBackend, RecoveryInput
from .graph_matern import _GraphMaternPosterior, _select_gp


class RBFGPBackend(RecoveryBackend):
    name = "rbf_gp"

    def __init__(
        self, *, lengthscale: float = 1.0, noise: float = 0.2, tune_noise: bool = True
    ) -> None:
        self.lengthscale = lengthscale
        self.noise = noise
        self.tune_noise = tune_noise
        self._posterior: _GraphMaternPosterior | None = None
        self._X: np.ndarray | None = None

    def _kernel(self, lengthscale: float) -> np.ndarray:
        X = self._X
        sq = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(X**2, axis=1)[None, :]
            - 2.0 * (X @ X.T)
        )
        return np.exp(-np.maximum(sq, 0.0) / (2.0 * lengthscale**2))

    def fit(self, recovery_input: RecoveryInput) -> None:
        if not recovery_input.features:
            raise RecoveryNotFitted("RBF GP requires RecoveryInput.features")
        node_ids = list(recovery_input.scientific_graph.node_ids)
        index = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        X = np.asarray(
            [recovery_input.features[nid] for nid in node_ids], dtype=float
        )
        # L2-normalize features (cosine-normalized) for scale-free RBF
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        self._X = X / norms

        obs_by_id = {o.object_id: o.value for o in recovery_input.observations}
        indices, values = [], []
        for oid, val in obs_by_id.items():
            if oid in index and np.isfinite(val):
                indices.append(index[oid])
                values.append(float(val))

        if not indices:
            lengthscale, noise = self.lengthscale, self.noise
            mean = np.zeros(n)
            cov = self._kernel(lengthscale)
        else:
            offset = float(np.mean(values))
            residuals = np.asarray(values) - offset
            if self.tune_noise:
                lengthscale, noise = _select_gp(
                    self._kernel, (0.25, 0.5, 1.0, 2.0, 4.0), indices, residuals,
                    self.lengthscale, self.noise,
                )
            else:
                lengthscale, noise = self.lengthscale, self.noise
            self._tuned_lengthscale = lengthscale
            self._tuned_noise = noise
            prior = self._kernel(lengthscale)
            k_oo = prior[np.ix_(indices, indices)]
            system = k_oo + np.eye(len(indices)) * noise**2
            k_xo = prior[:, indices]
            mean = offset + k_xo @ np.linalg.solve(system, residuals)
            cov = prior - k_xo @ np.linalg.solve(system, k_xo.T)
            cov = (cov + cov.T) / 2.0

        self._posterior = _GraphMaternPosterior(
            node_ids, index, mean, cov, recovery_input.solution_threshold
        )

    def posterior(self) -> PosteriorView:
        if self._posterior is None:
            raise RecoveryNotFitted("RBFGPBackend.fit() must be called first")
        return self._posterior

    def diagnostics(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "lengthscale": getattr(self, "_tuned_lengthscale", self.lengthscale),
            "noise": getattr(self, "_tuned_noise", self.noise),
        }
