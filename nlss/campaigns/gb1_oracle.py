"""GB1 finite graded oracle (evaluator-only) for the campaign runner (§10).

Wraps the real Wu-2016 GB1 measured fitness table (149,361 variants; 10,639
imputed NEVER merged in, §10.2 FROZEN).  The recovery **method** may only call
``evaluate(variant)`` within budget; gold accessors (``gold_of`` /
``solution_membership`` / ``gold_values`` / ``solution_set``) are evaluator-only.

Solution criteria (§10.4):
  * ``better_wt`` (PRIMARY): S*_WT = {x : f(x) > f(WT)}, WT = VDGV (|S*|=3,643).
  * ``top_pct``  (fallback):  S* = top-5% of measured fitness (|S*|=7,469).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..adapters.gb1.data import GB1Data, WILD_TYPE


class GB1FiniteOracle:
    def __init__(self, data: GB1Data | None = None, *, criterion: str = "better_wt",
                 top_frac: float = 0.05) -> None:
        if criterion not in ("better_wt", "top_pct"):
            raise ValueError(f"GB1 criterion must be better_wt|top_pct, got {criterion!r}")
        self.data = data if data is not None else GB1Data()
        self.criterion = criterion
        self.top_frac = top_frac
        self._fitness = dict(self.data.fitness)
        self._pool = list(self._fitness.keys())
        if criterion == "better_wt":
            self._gamma = float(self._fitness[WILD_TYPE])
            # strict > wild-type fitness
            self._sol = frozenset(v for v, f in self._fitness.items() if f > self._gamma)
        else:
            fs = np.asarray(list(self._fitness.values()))
            self._gamma = float(np.quantile(fs, 1.0 - top_frac))
            self._sol = frozenset(v for v, f in self._fitness.items() if f >= self._gamma)

    @property
    def candidates(self) -> Sequence[str]:
        return self._pool

    @property
    def gamma(self) -> float:
        return self._gamma

    def evaluate(self, variant: str) -> float:
        if variant not in self._fitness:
            raise ValueError(f"variant {variant!r} not in measured GB1 set (reject, no reveal)")
        return float(self._fitness[variant])

    def gold_of(self, variant: str) -> float:
        return float(self._fitness[variant])

    def solution_membership(self, variant: str) -> bool:
        return variant in self._sol

    @property
    def gold_values(self) -> Sequence[float]:
        return [self._fitness[v] for v in self._pool]

    @property
    def solution_set(self) -> Sequence[str]:
        return tuple(sorted(self._sol, key=lambda v: -self._fitness[v]))

    def solution_prevalence(self) -> float:
        return len(self._sol) / len(self._pool)

    @property
    def wild_type(self) -> str:
        return WILD_TYPE
