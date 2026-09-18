"""BH level-set active-learning challenger (report 9.8 B4, conceptual BALLET-style).

Attacks the claim "NLSS is just level-set / ROI recovery": a **dedicated**
region-of-interest learner that estimates the superlevel set S* = {x: f(x)>=gamma}
and actively queries the estimated boundary (uncertainty sampling), with no
structure beyond the raw feature space.

  * model   : L2-regularized logistic on component one-hot features
              -> P(sol) = P(f>=gamma | x), the level-set membership readout.
  * acquire : uncertainty sampling -- query the unqueried candidates whose
              predicted P(sol) is closest to 0.5 (refine the S* boundary).
  * recovery: score predict P(sol) with the same PR-AUC / RegionRecall / Brier

This is the strongest cheap level-set challenger; if NLSS materially beats it on
solution-space recovery, NLSS is not merely level-set identification.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .bh_oracle import BHFiniteOracle


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 60):
    """L2-regularized logistic regression by iterative reweighted least squares.

    Returns (weights, bias).  Handles K>2 (X already one-hot) and class imbalance
    by a small ridge on features.
    """
    N, D = X.shape
    Xb = np.hstack([X, np.ones((N, 1))])
    w = np.zeros(D + 1)
    reg = np.ones(D + 1) * l2
    reg[-1] = 0.0  # no penalty on bias
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        W = np.clip(p * (1 - p), 1e-9, 1e9)
        A = (Xb * W[:, None]).T @ Xb + np.diag(reg)
        grad = Xb.T @ (y - p)
        try:
            step = np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w[:-1], float(w[-1])


class BHBalletLevelSet:
    """Logistic level-set learner with uncertainty-sampling acquisition."""

    def __init__(self, oracle: BHFiniteOracle, features: Mapping[Any, np.ndarray]) -> None:
        self._pool = list(oracle.candidates)
        self._features = features
        self._gamma = oracle.gamma
        self._Xall = np.asarray([features[c] for c in self._pool], dtype=np.float64)
        self._idx = {c: i for i, c in enumerate(self._pool)}
        self._w: np.ndarray | None = None
        self._b = 0.0
        self._trained = False

    def fit(self, queried: Mapping[Any, float], train_seconds: list) -> None:
        import time

        t0 = time.time()
        idx = [self._idx[c] for c in queried if c in self._idx]
        if not idx:
            self._trained = False
            train_seconds.append(time.time() - t0)
            return
        X = self._Xall[idx]
        y = np.asarray([1.0 if queried[c] >= self._gamma else 0.0 for c in queried if c in self._idx])
        # guard: if a class is empty, still fit (will be constant model -> ties broken by margin)
        self._w, self._b = _fit_logistic(X, y, l2=1.0)
        self._trained = True
        train_seconds.append(time.time() - t0)

    @property
    def trained(self) -> bool:
        return self._trained

    def _p_all(self, pool_idx) -> np.ndarray:
        if self._w is None:
            return np.full(len(pool_idx), 0.5, dtype=np.float64)
        return _sigmoid(self._Xall[list(pool_idx)] @ self._w + self._b)

    def predict_all_solution_prob(self, candidates: Sequence[Any]) -> dict[Any, float]:
        idx = [self._idx[c] for c in candidates if c in self._idx]
        ps = self._p_all(idx)
        return {c: float(ps[i]) for i, c in enumerate(candidates) if self._idx.get(c) is not None}

    def acquire(self, queried, n: int, rng, coverage: float = 0.0) -> list[Any]:
        """BALLET-style superlevel-set active learning.

        Base: uncertainty sampling (query the boundary, |P-0.5| smallest).  With
        coverage>0, add a region-expansion reward: prefer boundary candidates
        whose one-factor neighbourhood contains many as-yet-unclaimed high-P
        candidates, expanding the estimated superlevel-set region (BALLET's
        coverage-reward acquisition, ICML 2023).
        """
        avail = [c for c in self._pool if c not in queried]
        if not avail:
            return []
        if not self._trained:
            rng.shuffle(avail)
            return avail[:n]
        idx = [self._idx[c] for c in avail]
        p = self._p_all(idx)
        margin = np.abs(p - 0.5)
        if coverage > 0:
            # region-expansion: for each candidate, count unqueried one-factor
            # neighbours already classified positive (P>=0.5) -> claims region
            unq = set(avail)
            unq_pos = {c for c in avail if p[avail.index(c)] >= 0.5}
            exp = np.zeros(len(avail))
            from collections import defaultdict
            d0 = sorted({c[0] for c in self._pool}); d1 = sorted({c[1] for c in self._pool})
            d2 = sorted({c[2] for c in self._pool}); doms = (d0, d1, d2, [c[3] for c in self._pool])
            for a_i, c in enumerate(avail):
                bonus = 0
                for k in range(4):
                    for x in doms[k]:
                        nc = c[:k] + (x,) + c[k+1:]
                        if nc in unq_pos:
                            bonus += 1
                exp[a_i] = bonus
            exp = exp / (exp.max() + 1e-9)
            score = margin - coverage * exp
            order = np.argsort(score)
        else:
            order = np.argsort(margin)
        return [avail[i] for i in order[:n]]
