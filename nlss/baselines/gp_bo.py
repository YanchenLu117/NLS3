"""V7 baseline — simple GP-BO (GP upper-confidence-bound Bayesian optimization).

Self-contained exact-RBF GP + UCB acquisition (mean + beta*std), used as a
SPECIALIST comparator in the controlled regime alongside GP-LSE.  Deterministic,
no external GP library (reuses the same numpy GP family as gp_lse).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class GPBandit:
    """GP-UCB over a candidate pool (self-contained exact-RBF)."""

    def __init__(
        self,
        *,
        beta: float = 2.0,
        sigma: float = 1.0,
        length_scale: float = 1.0,
        noise: float = 1e-3,
    ) -> None:
        self.beta = beta
        self.sigma = sigma
        self.length_scale = length_scale
        self.noise = noise
        self._X = np.empty((0, 1))
        self._y = np.empty(0)

    @staticmethod
    def _rows(arr) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        return a.reshape(-1, 1) if a.ndim == 1 else a

    def _k(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return self.sigma * np.exp(-0.5 * sq / (self.length_scale ** 2))

    def update(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> "GPBandit":
        X = self._rows(xs)
        self._X = np.vstack([self._X, X]) if len(self._X) else X
        self._y = np.concatenate([self._y, np.asarray(ys, dtype=float)])
        return self

    def select(self, pool: Sequence[Sequence[float]], ids: Sequence[str] | None = None, k: int = 1) -> tuple[tuple[str, ...], tuple[float, ...]]:
        X = self._rows(pool)
        if len(self._y) == 0:
            # no data yet: acquire by pure exploration (largest prior variance ~ uniform)
            score = np.full(len(X), 1.0)
        else:
            K = self._k(self._X, self._X) + self.noise * np.eye(len(self._X))
            Kinv = np.linalg.inv(K)
            Ks = self._k(self._X, X)
            mean = (Ks.T @ Kinv @ self._y)
            var = np.clip(np.diag(self._k(X, X)) - np.einsum("ij,ji->i", Ks.T @ Kinv, Ks), 0.0, None)
            score = mean + self.beta * np.sqrt(var)  # UCB
        order = np.argsort(-score, kind="stable")[:k]
        chosen = tuple(ids[i] for i in order) if ids else tuple(str(i) for i in order)
        return chosen, tuple(float(score[i]) for i in order)

    def info(self) -> dict[str, Any]:
        return {"name": "GP-BO (GP-UCB)", "beta": self.beta, "kernel": "exact-RBF"}
