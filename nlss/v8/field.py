"""V8 — attached generalized solution field (methods §3.3, Proposal §8).

The GSF attaches a probabilistic solution field to the computable space:

    r_t : Z_t ⇀ X_τ                         (response model, task-fixed kernel)
    B_t(y|z) = P(Y(z) = y | D_t, P_t)       (posterior over observations)
    p^sat_t(z) = E_{B_t}[ Sat(h_z, r_t(z), y) ]   ← EXPECTED SATISFACTION
    Ŝ_C,t(η) = { z ∈ dom(r_t) : p^sat_t(z) ≥ η }

p^sat is an EXPECTED SATISFACTION.  It is a posterior solution probability
only under a binary satisfaction indicator — the naming and the artifact
semantics keep this distinction explicit (Ω_NL is not Δ([0,1])).
``dom(r_t)`` 外的候选是 realization gap（无 r_t 值即无可执行化）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SolutionFieldPoint:
    z_id: str
    h_id: str
    p_sat: float  # expected satisfaction ∈ [0, 1]
    posterior: Mapping[str, float]  # B_t(y|z) — the distribution used
    in_domain: bool = True  # False ⇔ realization gap (z ∉ dom(r_t))


class SolutionFieldError(ValueError):
    """Field construction violates its contract."""


def expected_satisfaction(
    *,
    z_id: str,
    h_id: str,
    posterior: Mapping[str, float],
    response: Any,
    satisfaction_fn: Callable[[str, Any, str], float],
) -> SolutionFieldPoint:
    """p^sat_t(z) = Σ_y B_t(y|z) · Sat(h_z, r_t(z), y).

    ``posterior`` maps outcome y → probability (need not sum exactly to 1 —
    the missing mass is the model's residual; the caller's calibration audit
    owns it).  ``satisfaction_fn(h_id, response, y)`` returns Sat ∈ [0, 1].
    """
    total = 0.0
    mass = 0.0
    for y, prob in posterior.items():
        p = float(prob)
        if p < 0.0:
            raise SolutionFieldError("posterior probabilities must be non-negative")
        sat = float(satisfaction_fn(h_id, response, y))
        if not 0.0 <= sat <= 1.0:
            raise SolutionFieldError(f"Sat must be in [0,1] (got {sat} for y={y!r})")
        total += p * sat
        mass += p
    if mass <= 0.0:
        raise SolutionFieldError("posterior has non-positive mass")
    return SolutionFieldPoint(z_id=z_id, h_id=h_id, p_sat=total / mass, posterior=dict(posterior))


class SolutionField:
    """Ŝ_C,t(η) readouts over the attached field."""

    def __init__(self, points: Mapping[str, SolutionFieldPoint]) -> None:
        self._points = dict(points)

    def point(self, z_id: str) -> SolutionFieldPoint:
        return self._points[z_id]

    def solution_set(self, eta: float) -> frozenset[str]:
        """Ŝ_C,t(η) = { z : p^sat_t(z) ≥ η } (expected-satisfaction threshold)."""
        if not 0.0 <= eta <= 1.0:
            raise SolutionFieldError("eta must lie in [0,1]")
        return frozenset(z for z, pt in self._points.items() if pt.p_sat >= eta)

    def domain(self) -> frozenset[str]:
        return frozenset(self._points)

    def realization_gaps(self, candidate_ids: set[str]) -> frozenset[str]:
        """Candidates outside dom(r_t): represented hypotheses with no legal
        executable state (methods §3.9 realization gap)."""
        return frozenset(candidate_ids - set(self._points))

    def boundary(self, eta_low: float, eta_high: float) -> frozenset[str]:
        """Candidates in the η-band — the likely boundary readout class."""
        if eta_low > eta_high:
            raise ValueError("eta_low must be <= eta_high")
        return frozenset(
            z for z, pt in self._points.items() if eta_low <= pt.p_sat <= eta_high
        )

    def untested_alternatives(self, queried: set[str]) -> frozenset[str]:
        """High-saturation candidates never queried (inferred_untested readout)."""
        return frozenset(z for z, pt in self._points.items() if pt.p_sat >= 0.5 and z not in queried)
