"""GB1 pool-level recovery predictors (§10).

Two predictors over the 149,361 measured variants:

  * ``GB1HammingProp`` — Laplacian mean/confidence propagation over the FROZEN
    Hamming-1 mutation graph (§10.5).  A variant is adjacent to every other
    variant differing by exactly one amino acid at one of the 4 sites.  Observed
    fitness propagates to Hamming-1 neighbors, so an unqueried single-mutant of
    an observed high-fitness variant gets elevated predicted mean — the
    sparse->dense recovery mechanism for a protein landscape with strong
    epistasis.  Unlike BH, the Hamming-1 graph over S*_WT has meaningful
    (3) connected components, so RegionRecall is discriminating here.
  * ``GB1OneHotGP`` — naive feature RBF GP over the 80-d one-hot encoding
    (control: does graph structure add anything beyond a Euclidean kernel?).

The Hamming-1 neighbor index is outcome-blind (variant identity only) and built
once per process via a module cache (hash-lookup of the 76 possible single-mut
neighbors of each variant), so it is feasible at n=149,361.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..recovery.graph_matern import _norm_cdf, _select_gp

from .gb1_oracle import GB1FiniteOracle

_AA = "ACDEFGHIKLMNPQRSTVWY"
_N_SITES = 4


def _neighbors_of(variant: str) -> list[str]:
    """All 4*19=76 Hamming-1 neighbors of a 4-letter variant."""
    out = []
    for i in range(_N_SITES):
        for aa in _AA:
            if aa != variant[i]:
                out.append(variant[:i] + aa + variant[i + 1:])
    return out


class _HammingIndex:
    """Outcome-blind Hamming-1 adjacency built once (module cache)."""

    _cache: dict[tuple[str, ...], dict[int, list[int]]] = {}

    @staticmethod
    def build(pool: Sequence[str]) -> dict[int, list[int]]:
        pool_key = tuple(pool)
        cached = _HammingIndex._cache.get(pool_key)
        if cached is not None:
            return cached
        pos = {v: i for i, v in enumerate(pool)}
        adj: dict[int, list[int]] = {}
        for i, v in enumerate(pool):
            nbrs = []
            for nv in _neighbors_of(v):
                j = pos.get(nv)
                if j is not None and j != i:
                    nbrs.append(j)
            adj[i] = nbrs
        _HammingIndex._cache[pool_key] = adj
        return adj


class GB1HammingProp:
    _label_cache: dict[tuple[str, ...], dict[int, int] | None] = {}

    def __init__(self, oracle: GB1FiniteOracle) -> None:
        self._pool = list(oracle.candidates)
        self._index = {v: i for i, v in enumerate(self._pool)}
        self._gamma = oracle.gamma
        self._neighbors = _HammingIndex.build(self._pool)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._trained = False

    def fit(self, queried: Mapping[str, float], train_seconds: list) -> None:
        import time

        t0 = time.time()
        n = len(self._pool)
        mean = np.zeros(n)
        votes = np.zeros(n)
        nab = self._neighbors
        for v, val in queried.items():
            i = self._index.get(v)
            if i is None:
                continue
            for j in nab.get(i, ()):
                if j != i:
                    mean[j] += float(val)
                    votes[j] += 1.0
        values = [float(v) for v in queried.values()]
        global_mean = float(np.mean(values)) if values else 0.0
        mean = np.where(votes > 0, mean / np.maximum(votes, 1.0), global_mean)
        base = float(np.std(values)) if len(values) > 1 else 1.0
        std = base / np.sqrt(np.maximum(votes, 0.0) + 1.0)
        self._mean = mean
        self._std = std
        self._trained = True
        train_seconds.append(time.time() - t0)

    @property
    def trained(self) -> bool:
        return self._trained

    def predict_all_solution_prob(self, candidates: Sequence[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for v in candidates:
            i = self._index.get(v)
            if not self._trained or i is None or self._mean is None:
                out[v] = 0.5
                continue
            mu, sd = float(self._mean[i]), float(self._std[i])
            out[v] = float(mu >= self._gamma) if sd <= 0 else float(
                _norm_cdf((mu - self._gamma) / sd)
            )
        return out

    def predict_probs_by_index(self, indices: Sequence[int]) -> np.ndarray:
        """Vectorized p_t across pool indices (fast path for 149k-scale eval)."""
        if not self._trained or self._mean is None:
            return np.full(len(indices), 0.5, dtype=np.float64)
        mu = self._mean[np.asarray(indices, dtype=np.int64)]
        sd = self._std[np.asarray(indices, dtype=np.int64)]
        sol = np.empty_like(mu)
        pos = sd > 0
        sol[pos] = _npcdf((mu[pos] - self._gamma) / sd[pos])
        sol[~pos] = (mu[~pos] >= self._gamma).astype(np.float64)
        return sol

    @property
    def neighbor_count(self) -> int:
        return sum(len(nab) for nab in self._neighbors.values())
    @property
    def pool(self) -> list[str]:
        return self._pool

    @property
    def index(self) -> dict[str, int]:
        return self._index

    def component_labels(self) -> dict[str, int]:
        """Connected-component labels of the Hamming-1 graph over the POOL.

        NOTE: on the full 149,361 pool the Hamming-1 graph is one giant
        connected component (variant space is mutation-connected), so this only
        gives meaning when restricted to S* (the executor computes region
        membership over solution members).  We keep it for the single-region
        sanity and let the campaign compute RegionRecall over solution regions.
        """
        out: dict[str, int] = {}
        comp = [-1] * len(self._pool)
        cid = 0
        for start in range(len(self._pool)):
            if comp[start] != -1:
                continue
            stack = [start]
            comp[start] = cid
            while stack:
                u = stack.pop()
                for v in self._neighbors.get(u, ()):
                    if comp[v] == -1:
                        comp[v] = cid
                        stack.append(v)
            cid += 1
            if cid > 0:
                pass
        for v, i in self._index.items():
            out[v] = comp[i]
        return out


class GB1OneHotGP:
    """Naive RBF GP over the 80-d one-hot amino-acid encoding (control / GP-BO).

    Pure numpy implementation (proven stable for the nlss_onehot control arm).
    The full 149,361-variant pool is O(n_pool x n_train^2) per prediction pass,
    infeasible at full scale; ``max_pool_subset`` caps the arm's pool to a
    documented candidate subset.  ``acquire_ei`` adds Expected-Improvement
    acquisition so the same surrogate serves as a GP-BO arm (report 10.7).
    """

    def __init__(self, oracle: GB1FiniteOracle, max_pool_subset: int | None = None,
                 subset_seed: int = 0) -> None:
        self._pool = list(oracle.candidates)
        if max_pool_subset is not None and max_pool_subset < len(self._pool):
            rng = np.random.default_rng(subset_seed)
            keep = set(rng.choice(len(self._pool), size=max_pool_subset, replace=False).tolist())
            self._pool = [v for i, v in enumerate(self._pool) if i in keep]
        self._index = {v: i for i, v in enumerate(self._pool)}
        self._gamma = oracle.gamma
        self._X = np.asarray([self._onehot(v) for v in self._pool], dtype=np.float64)
        norms = np.linalg.norm(self._X, axis=1, keepdims=True) + 1e-9
        self._X = self._X / norms
        self._train_idx: list[int] = []
        self._Kinv = None
        self._alpha = None
        self._offset = 0.0
        self._l = 1.0
        self._noise = 0.4
        self._trained = False

    @staticmethod
    def _onehot(v: str) -> np.ndarray:
        vec = np.zeros(_N_SITES * len(_AA), dtype=np.float64)
        for i, ch in enumerate(v):
            vec[i * len(_AA) + _AA.index(ch)] = 1.0
        return vec

    def _kernel(self, Xa, Xb, l):
        sq = (np.sum(Xa**2, axis=1)[:, None] + np.sum(Xb**2, axis=1)[None, :] - 2.0 * (Xa @ Xb.T))
        return np.exp(-np.maximum(sq, 0.0) / (2.0 * l**2))

    def fit(self, queried: Mapping[str, float], train_seconds: list) -> None:
        import time

        t0 = time.time()
        idx = [self._index[v] for v in queried if v in self._index]
        if not idx:
            self._trained = False
            train_seconds.append(time.time() - t0)
            return
        Xin = self._X[idx]
        vals = np.asarray([queried[self._pool[i]] for i in idx], dtype=np.float64)
        offset = float(np.mean(vals))
        res = vals - offset
        l, noise = _select_gp(lambda p: self._kernel(Xin, Xin, p), (0.5, 1.0, 2.0, 4.0),
                              list(range(len(idx))), res, 1.0, 0.4)
        self._l, self._noise = l, noise
        self._offset = offset
        K = self._kernel(Xin, Xin, l)
        self._Kinv = np.linalg.inv(K + np.eye(len(idx)) * noise**2)
        self._alpha = self._Kinv @ res
        self._train_idx = idx
        self._trained = True
        train_seconds.append(time.time() - t0)

    @property
    def trained(self) -> bool:
        return self._trained

    def _posterior_at(self, pool_idx):
        """Return (mu, sd) numpy arrays at the given pool indices (RBF-GP posterior)."""
        idx = list(pool_idx)
        if not idx or not self._trained:
            return None, None
        Xin = self._X[self._train_idx]
        Kxt = self._kernel(self._X[idx], Xin, self._l)
        mu = self._offset + Kxt @ self._alpha
        var = 1.0 - np.einsum("ij,jk,ik->i", Kxt, self._Kinv, Kxt) + self._noise**2
        sd = np.sqrt(np.maximum(var, 0.0))
        return mu, sd

    def predict_all_solution_prob(self, candidates: Sequence[str]) -> dict[str, float]:
        if not self._trained:
            return {v: 0.5 for v in candidates}
        idx = [self._index[v] for v in candidates if v in self._index]
        if not idx:
            return {v: 0.5 for v in candidates}
        mu, sd = self._posterior_at(idx)
        sol = np.where(sd > 0, _npcdf((mu - self._gamma) / sd), (mu >= self._gamma).astype(float))
        return dict(zip([self._pool[i] for i in idx], sol.tolist()))

    def predict_probs_by_index(self, indices: Sequence[int]) -> np.ndarray:
        if not self._trained:
            return np.full(len(indices), 0.5, dtype=np.float64)
        idx = list(indices)
        mu, sd = self._posterior_at(idx)
        sol = np.where(sd > 0, _npcdf((mu - self._gamma) / sd), (mu >= self._gamma).astype(np.float64))
        return sol

    def acquire_ei(self, queried, n, rng=None):
        """Expected-Improvement acquisition (GP-BO arm, report 10.7 secondary)."""
        avail = [v for v in self._pool if v not in queried]
        if not avail:
            return []
        if not self._trained:
            if rng is not None:
                rng.shuffle(avail)
            return avail[:n]
        import math as _m
        idx = [self._index[v] for v in avail]
        mu, sd = self._posterior_at(idx)
        best = float(max(queried.values()))
        z = (mu - best) / np.maximum(sd, 1e-9)
        _sq2 = _m.sqrt(2.0)
        phi = np.exp(-0.5 * z * z) / _m.sqrt(2.0 * _m.pi)
        Phi = 0.5 * (1.0 + np.vectorize(_m.erf)(z / _sq2))
        ei = (mu - best) * Phi + sd * phi
        order = np.argsort(-ei)
        return [avail[i] for i in order[:n]]

    @property
    def pool(self) -> list[str]:
        return self._pool

    @property
    def index(self) -> dict[str, int]:
        return self._index


def _npcdf(x: np.ndarray) -> np.ndarray:
    from math import erf, sqrt

    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))
