"""Multi-scale diffusion Gaussian field backend (B4) — V6 primary candidate.

Follows the V6 experiment doc §2.4 exactly::

    K = sum_m alpha_m exp(-t_m L) + sigma_0^2 I

with a FIXED time bank ``t in {0.25, 1, 4, 16}`` and only the mixing weights
``alpha_m`` (>= 0, sum to 1) and the observation noise ``sigma_eps`` learned by
marginal likelihood.  Each base kernel ``exp(-t_m L)`` is correlation-normalized
(unit mean diagonal) so the prior variance is ~1 for the z-scored field.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.graph import laplacian_matrix
from ..core.types import PosteriorView
from .base import RecoveryBackend, RecoveryInput
from .graph_matern import _GraphMaternPosterior


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


class MultiScaleDiffusionBackend(RecoveryBackend):
    name = "multiscale_diffusion"

    def __init__(
        self,
        *,
        time_bank: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0),
        sigma_0: float = 0.05,
        noise: float = 0.2,
        tune: bool = True,
        n_restarts: int = 2,
    ) -> None:
        self.time_bank = tuple(time_bank)
        self.sigma_0 = sigma_0
        self.noise = noise
        self.tune = tune
        self.n_restarts = n_restarts
        self._posterior: _GraphMaternPosterior | None = None
        self._diag: dict[str, Any] = {}
        self._eig: tuple[np.ndarray, np.ndarray] | None = None
        self._base_kernels: list[np.ndarray] = []

    def _base_kernel(self, t: float) -> np.ndarray:
        eigenvalues, eigenvectors = self._eig
        spec = np.exp(-t * np.maximum(eigenvalues, 0.0))
        K = (eigenvectors * spec) @ eigenvectors.T
        K = (K + K.T) / 2.0
        d = float(np.mean(np.diag(K)))
        if d > 0:
            K = K / d
        return K

    def _kernel(self, alpha: np.ndarray) -> np.ndarray:
        K = np.zeros_like(self._base_kernels[0])
        for a, Km in zip(alpha, self._base_kernels):
            K = K + a * Km
        return K + self.sigma_0**2 * np.eye(K.shape[0])

    def _nll(self, logparams: np.ndarray, K_m_oo: list[np.ndarray], residuals: np.ndarray) -> float:
        m = len(self.time_bank)
        alpha = _softmax(logparams[:m])
        noise = float(np.exp(logparams[m]))
        Koo = self.sigma_0**2 * np.eye(len(residuals))
        for a, Kmo in zip(alpha, K_m_oo):
            Koo = Koo + a * Kmo
        system = Koo + noise**2 * np.eye(len(residuals))
        try:
            sign, logdet = np.linalg.slogdet(system)
            if sign <= 0:
                return 1e12
            solved = np.linalg.solve(system, residuals)
        except np.linalg.LinAlgError:
            return 1e12
        return 0.5 * (float(residuals @ solved) + float(logdet) + len(residuals) * np.log(2.0 * np.pi))

    def _learn(self, K_m_oo: list[np.ndarray], residuals: np.ndarray) -> tuple[np.ndarray, float]:
        from scipy.optimize import minimize

        m = len(self.time_bank)
        best = (np.inf, np.full(m, 1.0 / m), self.noise)
        for r in range(self.n_restarts):
            rng = np.random.default_rng(r)
            x0 = np.concatenate(
                [np.log(np.full(m, 1.0 / m)) + 0.1 * rng.normal(size=m), [np.log(self.noise)]]
            )
            res = minimize(
                self._nll, x0, args=(K_m_oo, residuals), method="L-BFGS-B",
                options={"maxiter": 40, "ftol": 1e-6},
            )
            if res.fun < best[0]:
                best = (res.fun, _softmax(res.x[:m]), float(np.exp(res.x[m])))
        return best[1], best[2]

    def fit(self, recovery_input: RecoveryInput) -> None:
        node_ids = list(recovery_input.scientific_graph.node_ids)
        if not node_ids:
            raise ValueError("scientific graph has no nodes")
        index = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        lap = laplacian_matrix(recovery_input.scientific_graph, node_ids=node_ids)
        self._eig = np.linalg.eigh(lap)
        self._base_kernels = [self._base_kernel(t) for t in self.time_bank]
        m = len(self.time_bank)

        obs_by_id = {o.object_id: o.value for o in recovery_input.observations}
        indices, values = [], []
        for oid, val in obs_by_id.items():
            if oid in index and np.isfinite(val):
                indices.append(index[oid])
                values.append(float(val))

        if not indices:
            alpha = np.full(m, 1.0 / m)
            noise = self.noise
            offset = 0.0
            mean = np.zeros(n)
            cov = self._kernel(alpha)
        else:
            offset = float(np.mean(values))
            residuals = np.asarray(values) - offset
            if self.tune:
                K_m_oo = [Km[np.ix_(indices, indices)] for Km in self._base_kernels]
                alpha, noise = self._learn(K_m_oo, residuals)
            else:
                alpha = np.full(m, 1.0 / m)
                noise = self.noise
            self._alpha = alpha
            self._tuned_noise = noise
            prior = self._kernel(alpha)
            k_oo = prior[np.ix_(indices, indices)]
            system = k_oo + np.eye(len(indices)) * noise**2
            k_xo = prior[:, indices]
            mean = offset + k_xo @ np.linalg.solve(system, residuals)
            cov = prior - k_xo @ np.linalg.solve(system, k_xo.T)
            cov = (cov + cov.T) / 2.0

        self._posterior = _GraphMaternPosterior(
            node_ids, index, mean, cov, recovery_input.solution_threshold
        )
        self._diag = {
            "n_nodes": n,
            "n_observations": len(indices),
            "time_bank": list(self.time_bank),
            "alpha": np.round(getattr(self, "_alpha", np.full(m, 1.0 / m)), 4).tolist(),
            "noise": getattr(self, "_tuned_noise", self.noise),
        }

    def posterior(self) -> PosteriorView:
        if self._posterior is None:
            from ..core.errors import RecoveryNotFitted

            raise RecoveryNotFitted("MultiScaleDiffusionBackend.fit() must be called first")
        return self._posterior

    def diagnostics(self) -> Mapping[str, Any]:
        return dict(self._diag)
