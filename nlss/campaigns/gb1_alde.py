"""ALDE (2020, active learning directed evolution) baseline for GB1.

Classic directed-evolution active learning (Sinai et al.): a GP surrogate over
observed (seq -> fitness), and each round acquisition proposes mutations of the
current best parent (and their combinations) scored by expected improvement,
querying the highest-EI.  Contrast with NLSS: ALDE optimizes fitness by active
learning; it does NOT produce an unqueried solution-space recovery readout.
Fidelity note: faithful ALDE protocol (GP+RBF, EI, mutant proposal from parent).
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import math as _m
import numpy as np
from nlss.campaigns.gb1_predict import GB1OneHotGP


class GB1Alde:
    """GP + EI directed evolution from a running best parent (ALDE protocol)."""

    def __init__(self, oracle, max_pool_subset=20000):
        self._gp = GB1OneHotGP(oracle, max_pool_subset=max_pool_subset)
        self._idx = self._gp.index
        self._pool = self._gp.pool
        self._gamma = oracle.gamma
        self._best = None
        self._top = []

    @property
    def index(self): return self._idx
    @property
    def pool(self): return self._pool

    def fit(self, queried: Mapping[str, float], train_seconds: list) -> None:
        if queried:
            self._best = max(queried, key=queried.get)
            self._top = [v for v, _ in sorted(queried.items(), key=lambda kv: -kv[1])[:10]]
        else:
            self._top = []
        self._gp.fit(queried, train_seconds)

    @property
    def trained(self): return self._gp.trained

    def predict_probs_by_index(self, indices: Sequence[int]) -> np.ndarray:
        return self._gp.predict_probs_by_index(indices)

    def predict_all_solution_prob(self, candidates) -> dict:
        return self._gp.predict_all_solution_prob(candidates)

    def _propose_parent_mutants(self, best, queried, cap=2000):
        """Single/two AA-substitution mutants of current parent(s) in the pool."""
        parents = self._top if getattr(self, "_top", None) else [best]
        out = []
        for par in parents:
            for i in range(len(par)):
                for aa in "ACDEFGHIKLMNPQRSTVWY":
                    cand = par[:i] + aa + par[i+1:]
                    if cand in self._idx and cand not in queried and cand not in out:
                        out.append(cand)
            if len(out) >= cap: break
        # double-mutants of the single best only
        if len(out) < cap:
            for i in range(len(best)):
                for j in range(i+1, len(best)):
                    for aa in "ACDEFGHIKLMNPQRSTVWY":
                        for bb in "ACDEFGHIKLMNPQRSTVWY":
                            cand = best[:i]+aa+best[i+1:j]+bb+best[j+1:]
                            if cand in self._idx and cand not in queried and cand not in out:
                                out.append(cand)
                            if len(out) >= cap: break
                        if len(out) >= cap: break
                    if len(out) >= cap: break
                if len(out) >= cap: break
        return out[:cap]
        for i in range(len(best)):            # single
            for aa in "ACDEFGHIKLMNPQRSTVWY":
                cand = best[:i] + aa + best[i+1:]
                if cand in self._idx and cand not in queried and cand not in out:
                    out.append(cand)
        for i in range(len(best)):            # double (limit for time)
            for j in range(i+1, len(best)):
                for aa in "ACDEFGHIKLMNPQRSTVWY":
                    for bb in "ACDEFGHIKLMNPQRSTVWY":
                        cand = best[:i]+aa+best[i+1:j]+bb+best[j+1:]
                        if cand in self._idx and cand not in queried and cand not in out:
                            out.append(cand)
                        if len(out) >= cap: break
                    if len(out) >= cap: break
                if len(out) >= cap: break
            if len(out) >= cap: break
        return out[:cap]

    def _ei_rank(self, proposals, queried):
        """Rank candidate variants by expected improvement of the GP posterior."""
        best = float(max(queried.values()))
        idx = [self._idx[v] for v in proposals]
        mu, sd = self._gp._posterior_at(idx)
        if mu is None:
            return list(range(len(proposals)))
        z = (mu - best) / np.maximum(sd, 1e-9)
        _sq2 = _m.sqrt(2.0)
        phi = np.exp(-0.5 * z * z) / _m.sqrt(2.0 * _m.pi)
        Phi = 0.5 * (1.0 + np.vectorize(_m.erf)(z / _sq2))
        ei = (mu - best) * Phi + sd * phi
        return np.argsort(-ei)

    def acquire(self, queried, n: int, rng, is_subset: bool):
        """ALDE: from the running best parent, propose single/double mutants and
        acquire the highest-expected-improvement; fall back to GP-EI over pool."""
        if not self.trained or self._best is None:
            avail = [v for v in self._pool if v not in queried]
            rng.shuffle(avail); return avail[:n]
        proposals = self._propose_parent_mutants(self._best, set(queried))
        if len(proposals) >= n:
            order = self._ei_rank(proposals, queried)
            return [proposals[i] for i in order[:n]]
        return self._gp.acquire_ei(queried, n, rng)
