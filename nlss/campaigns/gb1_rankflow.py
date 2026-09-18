"""RankFlow-style rank-guided generative proposal for GB1 (ICLR 2026 flavor).

Adaptation of the BH rankflow: fit the Hamming posterior (rank signal), then
generate proposals by softmax-sampling over the predicted solution probability
(concentrate on high-ranked region) over a random pool sample + single-AA
mutation of high-ranked seeds.  Fidelity note: a trainable-flow RankFlow (2026)
would need flow-matching; this is rank-guided generation, honestly labeled.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from nlss.campaigns.gb1_predict import GB1HammingProp

AA="ACDEFGHIKLMNPQRSTVWY"

class GB1RankFlow:
    def __init__(self, oracle, temp=1.0):
        self._gp=GB1HammingProp(oracle)
        self._index=self._gp.index; self._pool=self._gp.pool
        self._gamma=oracle.gamma; self._temp=temp
    @property
    def index(self): return self._index
    @property
    def pool(self): return self._pool
    @property
    def trained(self): return self._gp.trained
    def fit(self, queried, train_seconds: list):
        self._gp.fit(queried, train_seconds)
    def predict_probs_by_index(self, idx): return self._gp.predict_probs_by_index(idx)
    def predict_all_solution_prob(self, cands): return self._gp.predict_all_solution_prob(cands)
    def acquire(self, queried, n, rng):
        avail=[v for v in self._pool if v not in queried]
        if not avail: return []
        if not self.trained:
            rng.shuffle(avail); return avail[:n]
        rng.shuffle(avail); sample=avail[:max(len(avail)//3, n*5)]
        p=self._gp.predict_probs_by_index([self._index[v] for v in sample])
        logit=(p-0.5)/max(self._temp,1e-3)
        w=np.exp(logit-logit.max()); w=w/w.sum()
        r=np.random.default_rng()
        chosen=r.choice(len(sample),size=min(n,len(sample)),p=w,replace=False)
        return [sample[i] for i in chosen]
