"""Real generative-model acquisition (VAE-based GenBO, ICLR 2026 flavor) for BH.

A small categorical VAE over the 4-component BH space is trained on the observed
(top-fitness-weighted) candidates; acquisition SAMPLES from the VAE latent and
decodes candidate proposals (a genuine neural-generative acquisition sampler,
distinct from the llm_construct LLM-generator).  Fidelity note: a full GenBO
(2026) uses its upstream sampler; this is a real trained VAE generative
acquisition capturing the generative-acquisition mechanic.  GPU/CPU torch.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from .bh_oracle import BHFiniteOracle

AA="ACDEFGHIKLMNPQRSTVWY"  # unused
class _VAE(nn.Module):
    def __init__(self, xdim, zdim=8):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(xdim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.mu=nn.Linear(32,zdim); self.logv=nn.Linear(32,zdim)
        self.dec=nn.Sequential(nn.Linear(zdim,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,xdim))
    def forward(self,x):
        h=self.enc(x); mu=self.mu(h); lv=self.logv(h)
        z=mu+torch.randn_like(mu)*torch.exp(0.5*lv)
        return self.dec(z),mu,lv

class BHGenBO:
    def __init__(self, oracle, zdim=8, epochs=60, lr=1e-3):
        self._pool=list(oracle.candidates); self._idx={c:i for i,c in enumerate(self._pool)}
        self._gamma=oracle.gamma
        self._doms=[sorted({c[i] for c in self._pool}) for i in range(4)]
        self._xdim=sum(len(d) for d in self._doms)
        self._zdim=zdim; self._epochs=epochs; self._lr=lr
        self._model=_VAE(self._xdim,zdim)
        self._trained=False
    @property
    def trained(self): return self._trained
    def _encode(self,c):
        v=[]
        for i,d in enumerate(self._doms): v+=[1.0 if c[i]==x else 0.0 for x in d]
        return v
    def _decode_probe(self, out):
        # soft-prob -> pick argmax per block
        v=list(out); blocks=[]; k=0
        cand=[None]*4
        for i,d in enumerate(self._doms):
            sl=v[k:k+len(d)]; k+=len(d)
            j=int(np.argmax(sl)); cand[i]=d[j]
        return tuple(cand)
    def fit(self, queried: Mapping[Any,float], train_seconds:list):
        import time; t0=time.time()
        items=sorted(queried.items(),key=lambda kv:-kv[1])
        if len(items)<8: self._trained=False; train_seconds.append(time.time()-t0); return
        # top-80% fitness-weighted training set
        train=items[:max(8,int(len(items)*0.8))]
        X=np.asarray([self._encode(c) for c,_ in train],dtype=np.float32)
        xt=torch.tensor(X)
        dev='cuda' if torch.cuda.is_available() else 'cpu'; self._model=self._model.to(dev)
        opt=optim.Adam(self._model.parameters(),lr=self._lr)
        for _ in range(self._epochs):
            self._model.train(); opt.zero_grad()
            xb=xt.to(dev); rec,mu,lv=self._model(xb)
            recon=nn.functional.binary_cross_entropy_with_logits(rec,xb,reduction='sum')/xt.shape[0]
            kl=-0.5*torch.sum(1+lv-mu.pow(2)-lv.exp(),dim=1).mean()
            loss=recon+0.1*kl
            loss.backward(); opt.step()
        self._trained=True; train_seconds.append(time.time()-t0)
    def acquire(self, queried, n, rng):
        if not self._trained:
            avail=[c for c in self._pool if c not in queried]; rng.shuffle(avail); return avail[:n]
        self._model.eval(); seen=set(queried); out=[]
        trials=0; z=torch.randn(n*40,self._zdim,device=next(self._model.parameters()).device)
        with torch.no_grad():
            rec=self._model.dec(z).cpu().numpy()
        for r in rec:
            if len(out)>=n: break
            cand=self._decode_probe(r)
            if cand in self._idx and cand not in seen:
                out.append(cand); seen.add(cand)
        avail=[c for c in self._pool if c not in seen]; rng.shuffle(avail); out+=avail[:max(0,n-len(out))]
        return out[:n]
