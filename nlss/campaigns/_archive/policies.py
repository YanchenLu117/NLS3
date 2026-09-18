"""Baseline acquisition policies (NLSS V7 campaign).

Causal-control arms that consume the same oracle budget through one seam:

  * B0 — Random: uniform sampling from unqueried candidates (finite-oracle floor).
  * Hillclimb — outcome-blind single-factor greedy (structural neighbor climb).
    Not a paper baseline per se; a realistic "no-recovery-state" agent proxy used
    in the feasibility pilot for the headroom bracket.

LLM-hosted arms (B1-History / B2-Scalar) are wired in the BH campaign module so
they can share the same facades; the policies here are pure-computation.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from .base import AcquisitionPolicy, FiniteOracle


class RandomPolicy(AcquisitionPolicy):
    """B0 — uniform sampling over the unqueried candidate pool."""

    name = "random"

    def __init__(self, pool: Sequence[Any]) -> None:
        self._pool = list(pool)
        self._rng: random.Random | None = None

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def observe(self, candidate, outcome) -> None:
        pass

    def acquire(self, oracle: FiniteOracle, queried_cands: set[Any], n: int) -> list[Any]:
        avail = [c for c in self._pool if oracle.candidates and c not in queried_cands]
        if not self._pool and oracle.candidates:
            avail = [c for c in oracle.candidates if c not in queried_cands]
        self._rng.shuffle(avail)
        return avail[:n]


class GreedyHillclimbPolicy(AcquisitionPolicy):
    """Outcome-blind single-factor greedy (structural neighbor climb).

    Start from the best known candidate, propose its not-yet-queried one-factor
    neighbors, evaluate, keep climbing.  A realistic "no-recovery-state" agent
    that still exploits structure — used for the headroom bracket, NOT a paper
    baseline.
    """

    name = "hillclimb"

    def __init__(self, vocab: dict[str, Sequence[Any]], slots: Sequence[str]) -> None:
        self._vocab = vocab
        self._slots = list(slots)
        self._by_cand: dict[Any, float] = {}
        self._rng: random.Random | None = None
        self._initial_pool: list[Any] = []

    def set_initial_pool(self, pool: Sequence[Any]) -> None:
        self._initial_pool = list(pool)

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._by_cand = {}

    def observe(self, candidate, outcome) -> None:
        self._by_cand[candidate] = float(outcome)

    def acquire(self, oracle: FiniteOracle, queried_cands: set[Any], n: int) -> list[Any]:
        rng = self._rng
        pool = list(oracle.candidates)
        if not self._by_cand:
            # initial: random sample of n
            avail = [c for c in pool if c not in queried_cands]
            rng.shuffle(avail)
            return avail[:n]

        best = max(self._by_cand, key=lambda c: self._by_cand[c])
        proposed: list[Any] = []
        b = list(best)
        for slot_i, slot in enumerate(self._slots):
            for val in self._vocab[slot]:
                cand = tuple(b[:slot_i] + [val] + b[slot_i + 1:])
                if cand in pool and cand not in queried_cands:
                    proposed.append(cand)
        rng.shuffle(proposed)
        return proposed[:n]
