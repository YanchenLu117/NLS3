"""BH pool-level recovery predictor — RBF GP over outcome-blind one-hot features.

Trains on the **queried** observations only and predicts ``p_t(x) =
P(y(x) >= gamma | D_t)`` for any pool candidate, using the same RBF kernel +
marginal-likelihood noise/lengthscale tuning as the v7 ``rbf_gp`` backend
(reused from ``graph_matern``).  This is what lets the BH campaign evaluate
genuine *unseen* recovery over unqueried candidates (§9.9).

Efficiency: the kernel is only ever formed on the (small) training set and the
cross-covariance to the prediction set — never a full n_pool x n_pool matrix —
so the recovery loop stays cheap even with |S*| ~ 230 and a ~4k candidate pool.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..recovery.graph_matern import _norm_cdf, _select_gp

from .bh_oracle import BHFiniteOracle


def _rbf_kernel(Xa: np.ndarray, Xb: np.ndarray, lengthscale: float) -> np.ndarray:
    """RBF covariance between rows of Xa (n_a x d) and Xb (n_b x d)."""
    sq = (
        np.sum(Xa**2, axis=1)[:, None]
        + np.sum(Xb**2, axis=1)[None, :]
        - 2.0 * (Xa @ Xb.T)
    )
    return np.exp(-np.maximum(sq, 0.0) / (2.0 * lengthscale**2))


class BHPoolGP:
    """RBF-GP recovery predictor over the whole BH candidate pool.

    ``fit`` uses only queried (candidate, yield) pairs.  ``predict_solution_prob``
    returns P(sol | x) for any pool candidate.
    """

    def __init__(self, oracle: BHFiniteOracle, features: Mapping[Any, np.ndarray]) -> None:
        self._features = features
        self._gamma = oracle.gamma
        self._train_cands: list[Any] = []
        self._train_X: np.ndarray | None = None
        self._lengthscale: float = 1.0
        self._noise: float = 0.4
        self._offset: float = 0.0
        self._alpha: np.ndarray | None = None
        self._Kinv: np.ndarray | None = None
        self._std_train: np.ndarray | None = None
        self._trained = False

    @staticmethod
    def _norm(vec: np.ndarray) -> np.ndarray:
        return vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)

    def fit(self, queried: Mapping[Any, float], train_seconds: list) -> None:
        import time

        t0 = time.time()
        cands = [c for c in queried if c in self._features]
        if not cands:
            self._train_cands, self._train_X = [], None
            self._trained = False
            train_seconds.append(time.time() - t0)
            return
        X = self._norm(np.asarray([self._features[c] for c in cands]))
        vals = np.asarray([queried[c] for c in cands], dtype=float)

        offset = float(np.mean(vals))
        residuals = vals - offset
        prior_f = lambda ls: _rbf_kernel(X, X, ls)  # noqa: E731
        lengthscale, noise = _select_gp(prior_f, (0.25, 0.5, 1.0, 2.0, 4.0),
                                        list(range(len(cands))), residuals, 1.0, 0.4)
        self._lengthscale, self._noise = lengthscale, noise
        self._offset = offset
        k_oo = _rbf_kernel(X, X, lengthscale)
        system = k_oo + np.eye(len(cands)) * noise**2
        self._Kinv = np.linalg.inv(system)
        self._alpha = self._Kinv @ residuals
        self._train_cands = cands
        self._train_X = X
        # posterior training std (for consistency; unused for p_t directly)
        var = np.maximum(np.diag(np.ones(len(cands)) - k_oo @ self._Kinv), 0.0)
        self._std_train = np.sqrt(var + noise**2)
        self._trained = True
        train_seconds.append(time.time() - t0)

    @property
    def trained(self) -> bool:
        return self._trained

    def predict_solution_prob(self, candidate: Any) -> float:
        if not self._trained or candidate not in self._features:
            return 0.5
        x = self._norm(np.asarray([self._features[candidate]]))
        k_xo = _rbf_kernel(x, self._train_X, self._lengthscale)[0]  # (n_train,)
        mu = self._offset + float(k_xo @ self._alpha)
        var = 1.0 - float(k_xo @ self._Kinv @ k_xo) + self._noise**2
        sd = float(np.sqrt(max(var, 0.0)))
        if sd <= 0:
            return float(mu >= self._gamma)
        return float(_norm_cdf((mu - self._gamma) / sd))

    def predict_mean(self, candidate: Any) -> float:
        if not self._trained or candidate not in self._features:
            return self._offset
        x = self._norm(np.asarray([self._features[candidate]]))
        k_xo = _rbf_kernel(x, self._train_X, self._lengthscale)[0]
        return self._offset + float(k_xo @ self._alpha)

    def predict_all_solution_prob(self, candidates: Sequence[Any]) -> dict[Any, float]:
        """Batch prediction; normalizes all test features together (cheap)."""
        cands = [c for c in candidates if c in self._features]
        if not self._trained or not cands:
            return {c: 0.5 for c in candidates}
        Xt = self._norm(np.asarray([self._features[c] for c in cands]))
        Kxt = _rbf_kernel(Xt, self._train_X, self._lengthscale)  # (n_test, n_train)
        mu = self._offset + Kxt @ self._alpha
        var = 1.0 - np.einsum("ij,jk,ik->i", Kxt, self._Kinv, Kxt) + self._noise**2
        sd = np.sqrt(np.maximum(var, 0.0))
        sol = np.where(sd > 0, _npcdf((mu - self._gamma) / sd), (mu >= self._gamma).astype(float))
        return dict(zip(cands, sol.tolist()))

    @property
    def lengthscale(self) -> float:
        return self._lengthscale

    @property
    def noise(self) -> float:
        return self._noise


def _npcdf(x: np.ndarray) -> np.ndarray:
    from math import erf, sqrt

    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))
