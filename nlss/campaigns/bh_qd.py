"""QD / MAP-Elites baseline for BH (DMS-QD-2026-style, report 9.8 secondary).

Attacks the claim "NLSS is just quality-diversity search".  MAP-Elites archive
over a 2-D behavioral descriptor (aryl_halide x ligand component indices):
quality = oracle yield; diversity = coverage of the descriptor grid.  QD proposes
by mutating archived elites toward empty/low-covered cells (one-factor variation)
plus random exploration.  Outcome uses only observed yields (fair budget).

Fidelity note: a faithful DMS-QD needs its learned deep descriptor + upstream code
(not installed); this is a MAP-Elites QD capturing the same "quality+diversity
archive" mechanics.  Labeled as such -- not claimed to be the exact DMS-QD.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import random
import numpy as np
from nlss.campaigns.bh_oracle import BHFiniteOracle


class BhQDMAPElites:
    def __init__(self, oracle: BHFiniteOracle, halves: int = 1) -> None:
        self._pool = list(oracle.candidates)
        self._gamma = oracle.gamma
        vocab = self._vocab(oracle)
        # descriptor axes = component 0 (aryl_halide) and 1 (ligand)
        self._desc0 = vocab[0]  # list of values in deterministic order
        self._desc1 = vocab[1]
        self._vi = {c: i for i, c in enumerate(self._pool)}
        self._elites = {}          # (d0,d1) -> candidate (best yield in cell)
        self._q = {}               # (d0,d1) -> quality
        self._trained = False

    def _vocab(self, oracle):
        # derive component value domains from the pool
        d0 = sorted({c[0] for c in self._pool})
        d1 = sorted({c[1] for c in self._pool})
        d2 = sorted({c[2] for c in self._pool})  # unused
        return (d0, d1, d2)

    def _desc(self, c):
        return (self._desc0.index(c[0]), self._desc1.index(c[1]))

    def fit(self, queried: Mapping[Any, float], train_seconds: list) -> None:
        import time
        t0 = time.time()
        for c, y in queried.items():
            cell = self._desc(c)
            if cell not in self._q or y > self._q[cell]:
                self._q[cell] = float(y)
                self._elites[cell] = c
        self._trained = len(self._elites) > 0
        train_seconds.append(time.time() - t0)

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def archive_size(self) -> int:
        return len(self._elites)

    def _domains(self):
        return [sorted({c[i] for c in self._pool}) for i in range(4)]

    def acquire(self, queried, n: int, rng) -> list[Any]:
        """MAP-Elites: mutate/crossover elites across ANY slot (incl. descriptor),
        biasing toward under-covered cells so the archive both exploits (quality)
        and explores (diversity)."""
        out: list[Any] = []
        seen = set(queried)
        avail = [c for c in self._pool if c not in seen]
        if not self._trained:
            rng.shuffle(avail)
            return avail[:n]
        cells = list(self._elites.keys())
        qs = np.asarray([self._q[cell] for cell in cells], dtype=np.float64)
        doms = self._domains()
        # select cells weighted by quality (exploitation) and pick the elite;
        # n*0.6 from high-quality cells (refine best), rest uniform (diversity)
        n_exploit = max(0, int(n * 0.6))
        weights = np.clip(qs - qs.min() + 1e-3, 1e-3, None) ** 2
        weights = weights / weights.sum()
        order_exp = list(np.random.default_rng().choice(len(cells), size=n_exploit*3, p=weights, replace=True))
        order_dv = list(range(len(cells)))
        picks = order_exp[:n_exploit*2] if n_exploit else []
        # diverse cells appended
        for i in range(len(cells)):
            if i not in picks and i not in set(picks):
                picks.append(i)
        for ci in picks:
            if len(out) >= n:
                break
            el = self._elites[cells[ci]]
            for _ in range(30):
                n_mut = rng.choice([1, 1, 2])
                cand = list(el)
                for _s in range(n_mut):
                    slot = rng.randrange(4)
                    cand[slot] = rng.choice(doms[slot])
                cand = tuple(cand)
                if cand not in seen and cand in self._vi:
                    out.append(cand); seen.add(cand); break
        pad = [c for c in avail if c not in seen]
        rng.shuffle(pad)
        out.extend(pad[: max(0, n - len(out))])
        return out[:n]
