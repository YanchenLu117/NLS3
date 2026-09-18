"""RankFlow-style rank-guided generative proposal for BH (ICLR 2026 flavor).

RankFlow generates candidates from a rank/population distribution (rank-guided
generation).  Cheap, OOM-free adaptation: at each round fit a rank signal on
observed (here: one-factor graph posterior P(sol)), then GENERATE proposals by
sampling from a softmax over predicted rank (concentrating mass on high-ranked
region) crossed with one-factor mutation of high-ranked seeds.  Fidelity note:
trainable-flow RankFlow (2026) would need flow-matching over the discrete space;
this is a rank-guided generative proposal capturing the same "generate from the
ranked population" mechanic.  Labeled accordingly - not the exact upstream.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import random
import numpy as np
from nlss.campaigns.bh_graph import BHGraphProp


class BHRankFlow:
    def __init__(self, oracle, temp: float = 1.0) -> None:
        self._pool = list(oracle.candidates)
        self._idx = {c: i for i, c in enumerate(self._pool)}
        self._gamma = oracle.gamma
        self._temp = temp
        self._gp = BHGraphProp(oracle)
        self._trained = False

    @property
    def trained(self): return self._trained

    def fit(self, queried: Mapping[Any, float], train_seconds: list) -> None:
        import time
        t0 = time.time()
        self._gp.fit(queried, train_seconds)
        self._trained = self._gp.trained
        if not train_seconds:
            train_seconds.append(time.time() - t0)

    def predict_all_solution_prob(self, candidates: Sequence[Any]) -> dict[Any, float]:
        return self._gp.predict_all_solution_prob(candidates)

    def acquire(self, queried, n: int, rng) -> list[Any]:
        avail = [c for c in self._pool if c not in queried]
        if not avail:
            return []
        if not self._trained:
            rng.shuffle(avail); return avail[:n]
        pm = self._gp.predict_all_solution_prob(avail)
        p = np.asarray([pm[c] for c in avail])
        # rank-weighted generative: softmax over (p - 0.5) scaled by temp
        logit = (p - 0.5) / max(self._temp, 1e-3)
        w = np.exp(logit - logit.max())
        w = w / w.sum()
        rng2 = np.random.default_rng()
        chosen = rng2.choice(len(avail), size=n, p=w, replace=False)
        return [avail[i] for i in chosen]
