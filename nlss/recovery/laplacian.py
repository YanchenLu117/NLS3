"""Laplacian smoothing / label-propagation recovery backend (B1).

Deterministic graph propagation: each node's prediction is a weighted average of
its observed neighbors (one round of Laplacian smoothing).  This is the
non-parametric floor against which GP-style backends are compared.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..core.graph import adjacency_matrix
from ..core.types import MarginalPosterior, PairwiseDeltaPosterior, PosteriorView
from .base import RecoveryBackend, RecoveryInput


class _LaplacianPosterior:
    def __init__(
        self,
        node_ids: Sequence[str],
        index: dict[str, int],
        mean: np.ndarray,
        std: np.ndarray,
        threshold: float | None,
    ) -> None:
        self._node_ids = list(node_ids)
        self._index = index
        self._mean = mean
        self._std = std
        self._threshold = threshold

    def marginal(self, object_id: str) -> MarginalPosterior:
        i = self._index[object_id]
        std = float(self._std[i])
        from .graph_matern import _norm_cdf

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
        std_delta = float(np.sqrt(self._std[i] ** 2 + self._std[j] ** 2))
        from .graph_matern import _norm_cdf

        imp = float(_norm_cdf((mean_delta - delta) / std_delta)) if std_delta > 0 else float(mean_delta > delta)
        return PairwiseDeltaPosterior(
            source_id=source_id, target_id=target_id, mean_delta=mean_delta,
            std_delta=std_delta, improvement_probability=imp, threshold_delta=delta,
        )

    def sample_joint(
        self, object_ids: Sequence[str], n_samples: int, seed: int | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        idx = [self._index[oid] for oid in object_ids]
        return rng.normal(
            self._mean[idx], self._std[idx], size=(n_samples, len(idx))
        )


class LaplacianBackend(RecoveryBackend):
    name = "laplacian"

    def __init__(self, *, base_std: float | None = None) -> None:
        self.base_std = base_std
        self._posterior: _LaplacianPosterior | None = None

    def fit(self, recovery_input: RecoveryInput) -> None:
        node_ids = list(recovery_input.scientific_graph.node_ids)
        if not node_ids:
            raise ValueError("scientific graph has no nodes")
        index = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        adj = adjacency_matrix(recovery_input.scientific_graph, node_ids=node_ids)
        observed = {o.object_id: o.value for o in recovery_input.observations}

        mean = np.zeros(n)
        votes = np.zeros(n)
        for oid, val in observed.items():
            i = index.get(oid)
            if i is None:
                continue
            neighbors = np.nonzero(adj[i])[0]
            for j in neighbors:
                if j != i:
                    mean[j] += float(val)
                    votes[j] += 1.0
        # nodes with no observed neighbor fall back to global observed mean
        values = list(observed.values())
        global_mean = float(np.mean(values)) if values else 0.0
        mean = np.where(votes > 0, mean / np.maximum(votes, 1.0), global_mean)
        # base_std defaults to the observed-value spread so the uncertainty scale
        # matches the (z-scored) field; nodes with more observed neighbors are
        # sharper.
        base = self.base_std if self.base_std is not None else (
            float(np.std(values)) if len(values) > 1 else 1.0
        )
        self._base_std = base
        std = base / np.sqrt(np.maximum(votes, 0.0) + 1.0)

        self._posterior = _LaplacianPosterior(
            node_ids, index, mean, std, recovery_input.solution_threshold
        )

    def posterior(self) -> PosteriorView:
        if self._posterior is None:
            from ..core.errors import RecoveryNotFitted

            raise RecoveryNotFitted("LaplacianBackend.fit() must be called first")
        return self._posterior

    def diagnostics(self) -> Mapping[str, Any]:
        return {"backend": self.name, "base_std": getattr(self, "_base_std", self.base_std)}
