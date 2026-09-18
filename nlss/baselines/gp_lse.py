"""V7 baseline — GP-LSE (sequential level-set classification).

EXPERIMENT_PLAN §5: GP-LSE [Gotovos et al., IJCAI 2013] replaces BALLET (no
verifiable official impl).  The method uses GP confidence bounds to sequentially
classify a level set and expand informative regions.

This is a *deterministic, self-contained* numpy implementation (exact RBF GP,
no external GP library), used in the strict resource-matched controlled
comparison (C1/C2: Full NLSS vs GP-LSE on Recovery).  It is a documented VARIANT
of the original, not a claim to byte-exact reproduction:

Variants / differences (recorded here, per §5 requirement):
  - Original: implicit threshold / online quantile learning.  Here the level-set
    threshold is supplied explicitly (the specification repo exposes a
    solution-probability/observation threshold that is frozen on dev tasks).
  - Original is sequential; a batch of ``k`` is supported here for the
    resource-matched regime.
  - Acquisition is the boundary-straddling confidence interval: a point is
    informative when its posterior CI crosses the threshold
    (mean - beta*std <= threshold <= mean + beta*std); score = max(0,
    beta*std - |mean - threshold|), top-k selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class GPLSEDecision:
    """One acquisition step: selected ids + their level-set margin scores."""

    ids: tuple[str, ...]
    scores: tuple[float, ...]
    acquisition: str


class _ExactRBFGP:
    """Exact Gaussian-process regression with an RBF kernel (numpy)."""

    def __init__(
        self,
        sigma: float = 1.0,
        length_scale: float = 1.0,
        noise: float = 1e-3,
    ) -> None:
        self.sigma = sigma
        self.length_scale = length_scale
        self.noise = noise
        self.X = None
        self.y = None

    @staticmethod
    def _rows(arr) -> np.ndarray:
        """A 1-D pool is a list of scalar points -> (n,1); otherwise rows."""
        a = np.asarray(arr, dtype=float)
        return a.reshape(-1, 1) if a.ndim == 1 else a

    def _k(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return self.sigma * np.exp(-0.5 * sq / (self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ExactRBFGP":
        X = self._rows(X).astype(float)
        y = np.asarray(y, dtype=float).ravel()
        self.X, self.y = X, y
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._rows(X).astype(float)
        if self.X is None or len(self.X) == 0:
            n = X.shape[0]
            return np.zeros(n), np.full(n, self.sigma)
        K = self._k(self.X, self.X) + (self.noise * np.eye(len(self.X)))
        Kinv = np.linalg.inv(K)
        Ks = self._k(self.X, X)
        mean = Ks.T @ Kinv @ self.y
        var = np.diag(self._k(X, X)) - np.einsum("ij,ji->i", Ks.T @ Kinv, Ks)
        var = np.clip(var, 0.0, None)
        return mean, np.sqrt(var)

    @property
    def n(self) -> int:
        return 0 if self.X is None else len(self.X)


class GPLSE:
    """Sequential GP level-set classification baseline (documented variant)."""

    def __init__(
        self,
        threshold: float = 0.5,
        *,
        beta: float = 2.0,
        sigma: float = 1.0,
        length_scale: float = 1.0,
        noise: float = 1e-3,
    ) -> None:
        self.threshold = float(threshold)
        self.beta = beta
        self._gp = _ExactRBFGP(sigma=sigma, length_scale=length_scale, noise=noise)
        self._history: list[GPLSEDecision] = []

    def update(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> "GPLSE":
        self._gp.fit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        return self

    def predict(self, xs: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, std = self._gp.predict(np.asarray(xs, dtype=float))
        lo = mean - self.beta * std
        hi = mean + self.beta * std
        classify = np.where(hi < self.threshold, 0.0,
                            np.where(lo > self.threshold, 1.0, 0.5))  # 0.5 = uncertain/boundary
        return mean, std, classify

    def select(
        self,
        pool: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        k: int = 1,
    ) -> GPLSEDecision:
        """Pick the ``k`` most informative points (CI straddles the threshold)."""
        X = _ExactRBFGP._rows(pool).astype(float)
        mean, std = self._gp.predict(X)
        score = np.maximum(0.0, self.beta * std - np.abs(mean - self.threshold))
        order = np.argsort(-score, kind="stable")[:k]
        chosen = tuple(ids[i] for i in order) if ids else tuple(str(i) for i in order)
        dec = GPLSEDecision(ids=chosen, scores=tuple(float(score[i]) for i in order),
                            acquisition="boundary-ci-crossing")
        self._history.append(dec)
        return dec

    @property
    def n_queries(self) -> int:
        return sum(len(d.ids) for d in self._history)

    def info(self) -> dict[str, Any]:
        return {
            "name": "GP-LSE (Gotovos 2013 variant)",
            "threshold": self.threshold,
            "beta": self.beta,
            "kernel": "exact-RBF",
            "variant_notes": "explicit threshold (not online-quantile); batch-supported; "
                             "boundary-CI-crossing acquisition; exact numpy GP.",
        }
