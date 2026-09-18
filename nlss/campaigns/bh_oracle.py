"""BH finite graded oracle (evaluator-only) for the campaign runner.

Wraps the real 4,599-row ``BHData`` table.  The recovery **method** may only
call ``evaluate(candidate)`` within budget; gold accessors are evaluator-only
and never handed to an acquisition arm.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..adapters.bh.data import BHData, Candidate


class BHFiniteOracle:
    """BH candidate universe with hidden gold yields.

    ``evaluate`` only ever reveals the queried candidate's yield; it raises on a
    candidate absent from the measured table (reject-without-reveal, §9.3).  The
    gold accessors (``gold_of`` / ``solution_membership`` / ``gold_values`` /
    ``solution_set``) are for the evaluator only and MUST NOT be passed to an
    acquisition arm.
    """

    def __init__(self, data: BHData, *, top_frac: float = 0.05) -> None:
        self.data = data
        self.top_frac = top_frac
        self._gold = {c: y for c, y in data.records}
        # Candidates that can ground through the pipe-text interface (non-empty a|l|b).
        self._pool = [c for c, _ in data.records if all(str(x).strip() for x in c[:3])]
        ys = np.asarray([self._gold[c] for c in self._pool])
        self._gamma = float(np.quantile(ys, 1.0 - top_frac))
        self._sol = frozenset(c for c in self._pool if self._gold[c] >= self._gamma)

    @property
    def candidates(self) -> Sequence[Candidate]:
        return self._pool

    @property
    def gamma(self) -> float:
        return self._gamma

    def evaluate(self, candidate: Candidate) -> float:
        if candidate not in self._gold:
            raise ValueError(f"candidate {candidate!r} not in measured BH table (reject, no reveal)")
        return float(self._gold[candidate])

    def gold_of(self, candidate: Candidate) -> float:
        return float(self._gold[candidate])

    def solution_membership(self, candidate: Candidate) -> bool:
        return candidate in self._sol

    @property
    def gold_values(self) -> Sequence[float]:
        return [self._gold[c] for c in self._pool]

    @property
    def solution_set(self) -> Sequence[Candidate]:
        return tuple(sorted(self._sol, key=lambda c: -self._gold[c]))

    def solution_prevalence(self) -> float:
        return len(self._sol) / len(self._pool)
