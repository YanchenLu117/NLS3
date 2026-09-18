"""TPE (Bergstra et al., 2013) acquisition for BH (report 9.8 secondary anchor).

Tree-structured Parzen Estimator: model l(x) = P(x | good) and g(x) = P(x | bad)
via categorical Parzen densities over the component one-hot space; acquisition
maximizes the expected-improvement proxy l(x)/g(x) (Thompson-style).  Real TPE
algorithm, tuned quantile q=0.25.  Also exposes a recovery readout from the
density ratio (calibrated to [0,1]).
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import random, numpy as np
from nlss.campaigns.bh_oracle import BHFiniteOracle


def _sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z,-30,30)))

class BhTPE:
    def __init__(self, oracle, q=0.25, eps=0.1):
        self._pool=list(oracle.candidates); self._idx={c:i for i,c in enumerate(self._pool)}
        self._gamma=oracle.gamma; self._q=q; self._eps=eps
        # per-component domain
        self._domains=[sorted({c[i] for c in self._pool}) for i in range(4)]
        self._trained=False
    @property
    def trained(self): return self._trained
    def _parzen(self, pts):
        # pts: list of candidates -> per-dim smoothed categorical density maps
        # returns list of 4 dicts {value: count_prob} over the WHOLE domain
        dens=[]
        for dim in range(4):
            dom=self._domains[dim]
            cnt={v:0.0 for v in dom}
            for c in pts: cnt[c[dim]]+=1.0
            n=len(pts)
            # Parzen: smoothed = (count + eps*prior)/ (n + eps*len(dom))
            out={v:(cnt[v]+self._eps*(n/len(dom)))/(n+self._eps) for v in dom}
            dens.append(out)
        return dens
    def fit(self, queried: Mapping[Any,float], train_seconds: list):
        import time; t0=time.time()
        items=list(queried.items())
        if not items: self._trained=False; train_seconds.append(time.time()-t0); return
        items.sort(key=lambda kv:-kv[1])
        split=max(1,int(len(items)*self._q))
        good=[c for c,_ in items[:split]]; bad=[c for c,_ in items[split:]]
        self._lg=self._parzen(good); self._lg_prior=split
        self._gb=self._parzen(bad if bad else good)
        self._trained=True; train_seconds.append(time.time()-t0)
    def _logratio(self, cand):
        lr=0.0
        for dim in range(4):
            l=self._lg[dim].get(cand[dim],1e-9); g=self._gb[dim].get(cand[dim],1e-9)
            lr+=np.log((l+1e-9)/(g+1e-9))
        return lr
    def predict_all_solution_prob(self, candidates: Sequence[Any]) -> dict:
        out={}
        for c in candidates:
            lr=self._logratio(c) if self._trained else 0.0
            out[c]=float(_sigmoid(lr))  # calibrate density-ratio to prob
        return out
    def acquire(self, queried, n, rng):
        if not self._trained:
            avail=[c for c in self._pool if c not in queried]; rng.shuffle(avail); return avail[:n]
        avail=[c for c in self._pool if c not in queried]
        # sample a candidate subset and rank by log-ratio (TPE EI proxy); add noise for diversity
        rng.shuffle(avail)
        sample=avail[:max(len(avail)//2, n*4)]
        scores=[self._logratio(c) for c in sample]
        order=np.argsort(-np.array(scores))
        return [sample[i] for i in order[:n]]
