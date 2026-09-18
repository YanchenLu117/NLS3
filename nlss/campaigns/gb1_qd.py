"""QD / MAP-Elites for GB1 (DMS-QD-2026-flavor, report 10.7 secondary).

Behavioral descriptor = the 4 mutated sites (which of the 4 combinatorial loci
are non-WT / their substitution identities → the variant's site-identity vector).
Quality = fitness. Archive keeps best-per-cell; propose by mutating archives
elites (single AA substitution across any site) + random exploration.  Outcome
uses only observed fitness (fair budget).
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from nlss.campaigns.gb1_oracle import GB1FiniteOracle

AA="ACDEFGHIKLMNPQRSTVWY"

class GB1QDMAPElites:
    def __init__(self, oracle: GB1FiniteOracle):
        self._pool=list(oracle.candidates); self._idx={v:i for i,v in enumerate(self._pool)}
        self._gamma=oracle.gamma
        # realized WT at each of 4 sites (mode of pool) -> descriptor = (is_mut per site, which aa)
        self._wt=NULL=None
        self._elites={}; self._q={}; self._trained=False
    @property
    def index(self): return self._idx
    @property
    def pool(self): return self._pool
    def _desc(self, v):
        # descriptor = tuple of per-site (aa) ; archive cell = unique substitution vector
        return tuple(v)
    def fit(self, queried: Mapping[str,float], train_seconds: list):
        import time; t0=time.time()
        for c,y in queried.items():
            cell=self._desc(c)
            if cell not in self._q or y>self._q[cell]:
                self._q[cell]=float(y); self._elites[cell]=c
        self._trained=len(self._elites)>0
        train_seconds.append(time.time()-t0)
    @property
    def trained(self): return self._trained
    def acquire(self, queried, n:int, rng):
        out=[]; seen=set(queried)
        avail=[c for c in self._pool if c not in seen]
        if not self._trained:
            rng.shuffle(avail); return avail[:n]
        cells=list(self._elites.keys()); qs=np.asarray([self._q[x] for x in cells],float)
        w=np.clip(qs-qs.min()+1e-3,1e-3,None)**2; w=w/w.sum()
        r=np.random.default_rng()
        ne=max(0,int(n*0.6))
        picks=list(r.choice(len(cells),size=min(len(cells),ne*2),p=w,replace=True))
        picks+=list(range(len(cells)))
        for ci in picks:
            if len(out)>=n: break
            el=self._elites[cells[ci]]
            for _ in range(40):
                pos=rng.randrange(4); cand=el[:pos]+rng.choice(AA)+el[pos+1:]
                if cand in self._idx and cand not in seen:
                    out.append(cand); seen.add(cand); break
        rng.shuffle(avail); out+=[c for c in avail if c not in seen][:max(0,n-len(out))]
        return out[:n]
