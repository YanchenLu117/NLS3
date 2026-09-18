"""Real flow-matching generative proposer for BH (RankFlow-2026-flavor).

A genuine continuous flow-matching model: train a velocity network v_theta(x_t,t)
on the top-fitness candidates (one-hot/relaxed encoding) with the flow-matching
objective E ||v_theta - (x1-x0)||^2, then generate proposals by integrating the
ODE from noise and decoding argmax-per-component.  A real trained generative flow
(replacing the previous softmax proxy).  Fidelity note: a full RankFlow (2026)
uses its specific flow/ranking setup; this is a real flow-matching generator
capturing the generative-proposal mechanic.  GPU torch.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from .bh_oracle import BHFiniteOracle

class _VelNet(nn.Module):
    def __init__(self, xdim, zdim=16):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(xdim+1,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,xdim))
    def forward(self,x,t):
        return self.net(torch.cat([x,t.expand(x.shape[0],1)],dim=1))

class BHRankFlowReal:
    def __init__(self, oracle, zdim=16, epochs=40, lr=1e-3, steps=24):
        self._pool=list(oracle.candidates); self._idx={c:i for i,c in enumerate(self._pool)}
        self._gamma=oracle.gamma
        self._doms=[sorted({c[i] for c in self._pool}) for i in range(4)]
        self._xdim=sum(len(d) for d in self._doms)
        self._epochs=epochs; self._lr=lr; self._steps=steps
        self._model=_VelNet(self._xdim,zdim)
        self._trained=False
    @property
    def trained(self): return self._trained
    def _encode(self,c):
        v=[]
        for i,d in enumerate(self._doms): v+=[0.2 if c[i]!=x else 1.0 for x in d]
        return v
    def _decode(self, v):
        k=0; cand=[None]*4
        for i,d in enumerate(self._doms):
            j=int(np.argmax(v[k:k+len(d)])); cand[i]=d[j]; k+=len(d)
        return tuple(cand)
    def fit(self, queried, train_seconds:list):
        import time; t0=time.time()
        items=sorted(queried.items(),key=lambda kv:-kv[1])
        if len(items)<8: self._trained=False; train_seconds.append(time.time()-t0); return
        train=items[:max(8,int(len(items)*0.7))]
        X=np.asarray([self._encode(c) for c,_ in train],dtype=np.float32)
        xt=torch.tensor(X)
        dev='cuda' if torch.cuda.is_available() else 'cpu'; self._model=self._model.to(dev); self._dev=dev
        opt=optim.Adam(self._model.parameters(),lr=self._lr)
        for _ in range(self._epochs):
            self._model.train(); opt.zero_grad()
            t=torch.rand(len(xt),1,device=dev)
            x0=torch.randn_like(xt.to(dev))
            x1=xt.to(dev)
            x_t=(1-t)*x0 + t*x1
            vpred=self._model(x_t,t)
            loss=((vpred-(x1-x0))**2).mean()
            loss.backward(); opt.step()
        self._trained=True; train_seconds.append(time.time()-t0)
    def acquire(self, queried, n, rng):
        if not self._trained:
            avail=[c for c in self._pool if c not in queried]; rng.shuffle(avail); return avail[:n]
        self._model.eval(); dev=self._dev
        seen=set(queried); out=[]
        z=torch.randn(n*60,self._xdim,device=dev)
        with torch.no_grad():
            dt=1.0/self._steps
            x=z.clone()
            for s in range(self._steps):
                t=torch.full((x.shape[0],1),s*dt,device=dev)
                v=self._model(x,t)
                x=x+v*dt
        xc=x.cpu().numpy()
        for r in xc:
            if len(out)>=n: break
            cand=self._decode(r)
            if cand in self._idx and cand not in seen:
                out.append(cand); seen.add(cand)
        avail=[c for c in self._pool if c not in seen]; rng.shuffle(avail); out+=avail[:max(0,n-len(out))]
        return out[:n]
