"""Graph-Matérn GP recovery backend (V6 reference backend).

Covariance ``K = variance * (kappa^2 I + L)^(-nu)`` with Gaussian observation
noise.  Dense exact inference is suitable for contract tests and small graphs;
formal scale uses an equivalent low-rank/sparse path behind the same interface.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..core.graph import laplacian_matrix
from ..core.types import (
    MarginalPosterior,
    PairwiseDeltaPosterior,
    PosteriorView,
)
from .base import RecoveryBackend, RecoveryInput


class _GraphMaternPosterior:
    def __init__(
        self,
        node_ids: Sequence[str],
        index: dict[str, int],
        mean: np.ndarray,
        covariance: np.ndarray,
        solution_threshold: float | None,
    ) -> None:
        self._node_ids = list(node_ids)
        self._index = index
        self._mean = mean
        self._covariance = covariance
        self._threshold = solution_threshold

    def marginal(self, object_id: str) -> MarginalPosterior:
        i = self._index[object_id]
        std = float(np.sqrt(max(self._covariance[i, i], 0.0)))
        sol_prob = None
        if self._threshold is not None:
            sol_prob = float(_norm_cdf((self._mean[i] - self._threshold) / std)) if std > 0 else float(self._mean[i] >= self._threshold)
        return MarginalPosterior(
            object_id=object_id, mean=float(self._mean[i]), std=std, solution_probability=sol_prob
        )

    def pairwise_delta(
        self, source_id: str, target_id: str, delta: float = 0.0
    ) -> PairwiseDeltaPosterior:
        i = self._index[source_id]
        j = self._index[target_id]
        mean_delta = float(self._mean[j] - self._mean[i])
        var_delta = max(
            float(self._covariance[i, i] + self._covariance[j, j] - 2.0 * self._covariance[i, j]),
            0.0,
        )
        std_delta = float(np.sqrt(var_delta))
        imp = (
            float(_norm_cdf((mean_delta - delta) / std_delta))
            if std_delta > 0
            else float(mean_delta > delta)
        )
        return PairwiseDeltaPosterior(
            source_id=source_id,
            target_id=target_id,
            mean_delta=mean_delta,
            std_delta=std_delta,
            improvement_probability=imp,
            threshold_delta=delta,
        )

    def sample_joint(
        self, object_ids: Sequence[str], n_samples: int, seed: int | None = None
    ) -> np.ndarray:
        idx = [self._index[oid] for oid in object_ids]
        sub_mean = self._mean[idx]
        sub_cov = self._covariance[np.ix_(idx, idx)]
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(sub_mean, sub_cov, size=n_samples)


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _marginal_nll(
    prior: np.ndarray,
    indices: list[int],
    residuals: np.ndarray,
    noise: float,
) -> float:
    if not indices:
        return np.inf
    k_oo = prior[np.ix_(indices, indices)]
    system = k_oo + np.eye(len(indices)) * noise**2
    try:
        sign, logdet = np.linalg.slogdet(system)
        if sign <= 0:
            return np.inf
        solved = np.linalg.solve(system, residuals)
    except np.linalg.LinAlgError:
        return np.inf
    return 0.5 * (float(residuals @ solved) + float(logdet) + len(indices) * np.log(2.0 * np.pi))


def _select_gp(
    prior_fn,
    param_grid: tuple[float, ...],
    indices: list[int],
    residuals: np.ndarray,
    default_param: float,
    default_noise: float,
    noise_grid: tuple[float, ...] = (0.1, 0.4, 1.6),
) -> tuple[float, float]:
    """Jointly select (kernel smoothness param, observation noise) by marginal
    likelihood (V6 §2.5 "learn the kernel scale + sigma_eps")."""
    best = (np.inf, default_param, default_noise)
    for p in param_grid:
        try:
            prior = prior_fn(p)
        except Exception:
            continue
        for noise in noise_grid:
            nll = _marginal_nll(prior, indices, residuals, noise)
            if nll < best[0]:
                best = (nll, p, noise)
    return best[1], best[2]


def _select_noise(
    prior: np.ndarray,
    indices: list[int],
    residuals: np.ndarray,
    fallback: float,
) -> float:
    """Backward-compatible noise-only selector (fixed kernel)."""
    _, noise = _select_gp(
        lambda _p: prior, (fallback,), indices, residuals, fallback, fallback
    )
    return noise


class GraphMaternBackend(RecoveryBackend):
    name = "graph_matern"

    def __init__(
        self,
        *,
        kappa: float = 1.0,
        nu: float = 2.0,
        variance: float = 1.0,
        noise: float = 0.2,
        tune_noise: bool = True,
    ) -> None:
        self.kappa = kappa
        self.nu = nu
        self.variance = variance
        self.noise = noise
        self.tune_noise = tune_noise
        self._posterior: _GraphMaternPosterior | None = None
        self._offset = 0.0
        self._diag: dict[str, Any] = {}
        self._lap: np.ndarray | None = None
        self._eig: tuple[np.ndarray, np.ndarray] | None = None

    def _kernel(self, kappa: float) -> np.ndarray:
        eigenvalues, eigenvectors = self._eig
        spectrum = np.power(
            float(kappa) ** 2 + np.maximum(eigenvalues, 0.0), -float(self.nu)
        )
        prior = (eigenvectors * spectrum) @ eigenvectors.T
        prior = (prior + prior.T) / 2.0
        # correlation normalization: unit mean diagonal, amplitude = variance
        d = float(np.mean(np.diag(prior)))
        if d > 0:
            prior = prior / d
        return float(self.variance) * prior

    def fit(self, recovery_input: RecoveryInput) -> None:
        node_ids = list(recovery_input.scientific_graph.node_ids)
        if not node_ids:
            raise ValueError("scientific graph has no nodes")
        index = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        if self.kappa <= 0 or self.nu <= 0 or self.variance <= 0 or self.noise <= 0:
            raise ValueError("kappa, nu, variance, noise must be positive")
        lap = laplacian_matrix(recovery_input.scientific_graph, node_ids=node_ids)
        self._eig = np.linalg.eigh(lap)

        obs_by_id = {o.object_id: o.value for o in recovery_input.observations}
        indices: list[int] = []
        values: list[float] = []
        for oid, val in obs_by_id.items():
            if oid in index and np.isfinite(val):
                indices.append(index[oid])
                values.append(float(val))

        if not indices:
            self._offset = 0.0
            kappa, noise = self.kappa, self.noise
            mean = np.zeros(n)
            cov = self._kernel(kappa)
        else:
            self._offset = float(np.mean(values))
            residuals = np.asarray(values) - self._offset
            if self.tune_noise:
                kappa, noise = _select_gp(
                    self._kernel, (0.25, 0.5, 1.0, 2.0, 4.0), indices, residuals,
                    self.kappa, self.noise,
                )
            else:
                kappa, noise = self.kappa, self.noise
            self._tuned_kappa = kappa
            self._tuned_noise = noise
            prior = self._kernel(kappa)
            k_oo = prior[np.ix_(indices, indices)]
            system = k_oo + np.eye(len(indices)) * noise**2
            k_xo = prior[:, indices]
            weights = np.linalg.solve(system, residuals)
            mean = self._offset + k_xo @ weights
            cov = prior - k_xo @ np.linalg.solve(system, k_xo.T)
            cov = (cov + cov.T) / 2.0

        self._posterior = _GraphMaternPosterior(
            node_ids, index, mean, cov, recovery_input.solution_threshold
        )
        self._diag = {
            "n_nodes": n,
            "n_observations": len(indices),
            "offset": self._offset,
            "kappa": getattr(self, "_tuned_kappa", self.kappa),
            "noise": getattr(self, "_tuned_noise", self.noise),
        }

    def posterior(self) -> PosteriorView:
        if self._posterior is None:
            from ..core.errors import RecoveryNotFitted

            raise RecoveryNotFitted("GraphMaternBackend.fit() must be called first")
        return self._posterior

    def diagnostics(self) -> Mapping[str, Any]:
        return dict(self._diag)
