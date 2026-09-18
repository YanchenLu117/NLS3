"""GB1 level-set active-learning challenger (report 10.7 conceptual BALLET-style).

A dedicated Region-of-Interest learner on the protein landscape: learns the
superlevel set S* = {x: fitness(x) >= WT} from a binary logistic model over the
80-d one-hot encoding, and actively queries the estimated boundary (uncertainty
sampling).  Unlike the feature GP, the logistic is cheap over the FULL 149k pool
(no O(n_pool x n_train^2) kernel), so this challenger is not subset-limited.

  * model   : L2-logistic on one-hot -> P(sol)=P(fitness>=WT | x)
  * acquire : uncertainty sampling (|P(sol)-0.5| closest first)
  * recovery: score P(sol) with the same PR-AUC / RegionRecall@K / Brier
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from .gb1_oracle import GB1FiniteOracle

_AA = "ACDEFGHIKLMNPQRSTVWY"; _NS = 4

def _onehot(v):
    vec = np.zeros(_NS*len(_AA), dtype=np.float64)
    for i,ch in enumerate(v):
        vec[i*len(_AA)+_AA.index(ch)] = 1.0
    return vec

def _sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z,-30,30)))

def _fit_logistic(X, y, l2=1.0, iters=60):
    N,D=X.shape
    Xb=np.hstack([X, np.ones((N,1))]); w=np.zeros(D+1)
    reg=np.ones(D+1)*l2; reg[-1]=0.0
    for _ in range(iters):
        p=_sigmoid(Xb@w); W=np.clip(p*(1-p),1e-9,1e9)
        try:
            step=np.linalg.solve((Xb*W[:,None]).T@Xb+np.diag(reg), Xb.T@(y-p))
        except np.linalg.LinAlgError:
            break
        w=w+step
        if np.max(np.abs(step))<1e-6: break
    return w[:-1], float(w[-1])

class GB1BalletLevelSet:
    def __init__(self, oracle: GB1FiniteOracle) -> None:
        self._pool=list(oracle.candidates)
        self._idx={v:i for i,v in enumerate(self._pool)}
        self._gamma=oracle.gamma
        self._X=np.asarray([_onehot(v) for v in self._pool],dtype=np.float64)
        self._w=None; self._b=0.0; self._trained=False
    def fit(self, queried: Mapping[str,float], train_seconds: list) -> None:
        import time; t0=time.time()
        idx=[i for v,i in self._idx.items() if v in queried]
        if not idx:
            self._trained=False; train_seconds.append(time.time()-t0); return
        X=self._X[idx]; y=np.asarray([1.0 if queried[self._pool[i]]>=self._gamma else 0.0 for i in idx])
        self._w,self._b=_fit_logistic(X,y,l2=1.0); self._trained=True
        train_seconds.append(time.time()-t0)
    @property
    def trained(self)->bool: return self._trained
    def _p(self, idx):
        return _sigmoid(self._X[list(idx)]@self._w+self._b) if self._w is not None else np.full(len(idx),0.5)
    def predict_probs_by_index(self, indices: Sequence[int])->np.ndarray:
        return self._p(list(indices))
    def predict_all_solution_prob(self, candidates)->dict:
        idx=[self._idx[v] for v in candidates if v in self._idx]
        ps=self._p(idx)
        return {v: float(ps[i]) for i,v in enumerate(candidates) if self._idx.get(v) is not None}
    def acquire(self, queried, n, rng):
        avail=[v for v in self._pool if v not in queried]
        if not avail: return []
        if not self._trained: rng.shuffle(avail); return avail[:n]
        idx=[self._idx[v] for v in avail]
        p=self._p(idx); margin=np.abs(p-0.5)
        order=np.argsort(margin)
        return [avail[i] for i in order[:n]]
    @property
    def pool(self): return self._pool
    @property
    def index(self): return self._idx
