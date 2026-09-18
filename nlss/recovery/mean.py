"""Mean / observed-only recovery backend (B0) — the trivial floor.

Predicts the global mean of observed values for every node, with uncertainty
from the observed-value spread.  Used as the bake-off floor so every other
backend's added value is measured against a non-informative baseline.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..core.types import MarginalPosterior, PairwiseDeltaPosterior, PosteriorView
from .base import RecoveryBackend, RecoveryInput


class _MeanPosterior:
    def __init__(
        self,
        node_ids: Sequence[str],
        index: dict[str, int],
        mean: float,
        std: float,
        threshold: float | None,
    ) -> None:
        self._node_ids = list(node_ids)
        self._index = index
        self._mean = mean
        self._std = std
        self._threshold = threshold

    def marginal(self, object_id: str) -> MarginalPosterior:
        from .graph_matern import _norm_cdf

        sol_prob = None
        if self._threshold is not None:
            sol_prob = float(_norm_cdf((self._mean - self._threshold) / self._std)) if self._std > 0 else float(self._mean >= self._threshold)
        return MarginalPosterior(object_id=object_id, mean=self._mean, std=self._std, solution_probability=sol_prob)

    def pairwise_delta(
        self, source_id: str, target_id: str, delta: float = 0.0
    ) -> PairwiseDeltaPosterior:
        from .graph_matern import _norm_cdf

        std = float(np.sqrt(2.0) * self._std)
        imp = float(_norm_cdf((0.0 - delta) / std)) if std > 0 else 0.5
        return PairwiseDeltaPosterior(
            source_id=source_id, target_id=target_id, mean_delta=0.0,
            std_delta=std, improvement_probability=imp, threshold_delta=delta,
        )

    def sample_joint(
        self, object_ids: Sequence[str], n_samples: int, seed: int | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(self._mean, self._std, size=(n_samples, len(object_ids)))


class MeanBackend(RecoveryBackend):
    name = "mean"

    def __init__(self) -> None:
        self._posterior: _MeanPosterior | None = None

    def fit(self, recovery_input: RecoveryInput) -> None:
        node_ids = list(recovery_input.scientific_graph.node_ids)
        values = [o.value for o in recovery_input.observations if np.isfinite(o.value)]
        mean = float(np.mean(values)) if values else 0.0
        std = float(np.std(values)) if len(values) > 1 else 1.0
        index = {nid: i for i, nid in enumerate(node_ids)}
        self._posterior = _MeanPosterior(
            node_ids, index, mean, max(std, 1e-6), recovery_input.solution_threshold
        )

    def posterior(self) -> PosteriorView:
        if self._posterior is None:
            from ..core.errors import RecoveryNotFitted

            raise RecoveryNotFitted("MeanBackend.fit() must be called first")
        return self._posterior

    def diagnostics(self) -> Mapping[str, Any]:
        return {"backend": self.name}
