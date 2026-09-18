"""V7 Core — Exploration Controller (acquistion-layer dial, proposal §17).

Implements the design in notes/EXPLORATION_CONTROLLER_V0_1.md, unified with the
five user-named gears:

    OFF / LOW / MEDIUM / HIGH / ULTRA

It is placed in the CORE (benchmark-agnostic), NOT the adapter layer.  The
controller decides *where NLSS proposes next* without touching the evidence
ledger, the logic/fidelity gates, the oracle budget, or the posterior model.
It operates over proposal policies rather than raw acquisition scores, so
channels with incompatible numerical scales (p_sol vs uncertainty vs coverage
gap) never fight each other.

Four proposal channels:
    SOL  (exploit)  high posterior solution probability / expected utility
    BND  (boundary) uncertain states near the current solution threshold eta
    COV  (coverage) under-covered regions & represented peripheries
    OPEN (open)     language-side hypotheses that may leave dom(Gamma_t); an
                    ungroundable proposal becomes a representation challenge
                    instead of receiving a fabricated field score.

A gear selects a channel first, then a candidate is sampled from that
channel's *normalized* policy:

    J_t ~ Categorical(w_g),   a_t ~ pi_t^{J_t}

Paper-level boundary (v0.1 §"Paper-level claim boundary"): NLSS does not claim
a new universal acquisition function.  It claims that exploration is controlled
over a progressively constructed scientific solution-space state, while the
language track keeps an explicit route beyond the current computable space.
Actually learning an optimal gear policy is OUT of the core claim; the gear is
an explicit, intervenable control (with an optional recommendation helper).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class ExplorationGear(str, Enum):
    """Five exploration gears.

    ``OFF`` disables exploratory proposal allocation (pure exploitation) and is
    the default, preserving the original V7 behavior exactly.  Off does NOT
    disable logic/fidelity monitoring, evidence ingestion, or a mandatory
    representation-revision response when the representation is already known
    invalid (v0.1).
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


# Channel constants (stable order used everywhere: SOL, BND, COV, OPEN).
SOL, BND, COV, OPEN = ("SOL", "BND", "COV", "OPEN")
CHANNELS: tuple[str, ...] = (SOL, BND, COV, OPEN)


@dataclass(frozen=True, slots=True)
class GearProfile:
    """One gear = a categorical distribution over the four channels.

    `w_sol / w_bnd / w_cov / w_open` sum to 1.  Exploration mass is
    ``1 - w_sol`` and the ``OPEN`` mass both increase monotonically from
    OFF to ULTRA (v0.1).  ``tau_b`` (boundary temperature) and ``beta_c``
    (coverage-uncertainty mixing) are within-channel tuning, not gear exclusions.
    """

    gear: ExplorationGear
    paper_name: str
    w_sol: float
    w_bnd: float
    w_cov: float
    w_open: float
    tau_b: float = 0.1
    beta_c: float = 1.0

    @property
    def weights(self) -> tuple[float, float, float, float]:
        return (self.w_sol, self.w_bnd, self.w_cov, self.w_open)

    @property
    def exploration_mass(self) -> float:
        return 1.0 - self.w_sol

    def weight_of(self, channel: str) -> float:
        return {"SOL": self.w_sol, "BND": self.w_bnd, "COV": self.w_cov,
                "OPEN": self.w_open}[channel]


# Recommended defaults (PILOT-FREEZE before final evaluation; v0.1 table,
# gear names mapped to the user's five-level naming).
_DEFAULT_PROFILES: dict[ExplorationGear, GearProfile] = {
    ExplorationGear.OFF: GearProfile(ExplorationGear.OFF, "exploit-only", 1.00, 0.00, 0.00, 0.00),
    ExplorationGear.LOW: GearProfile(ExplorationGear.LOW, "conservative", 0.65, 0.20, 0.10, 0.05),
    ExplorationGear.MEDIUM: GearProfile(ExplorationGear.MEDIUM, "balanced", 0.45, 0.25, 0.20, 0.10),
    ExplorationGear.HIGH: GearProfile(ExplorationGear.HIGH, "frontier", 0.20, 0.25, 0.35, 0.20),
    ExplorationGear.ULTRA: GearProfile(ExplorationGear.ULTRA, "open-world", 0.10, 0.15, 0.25, 0.50),
}


class ExplorationController:
    """Benchmark-agnostic acquisition-layer dial.

    Parameters
    ----------
    coverage_gap:
        Optional ``oid -> float`` callable returning a normalized coverage-gap
        score for ``COV``.  When None, coverage-gap falls back to posterior
        uncertainty (under-covered ~ high-uncertainty), which needs no
        benchmark geometry and keeps the controller fully generic.
    """

    def __init__(
        self,
        *,
        coverage_gap: Callable[[str], float] | None = None,
        profiles: Mapping[ExplorationGear, GearProfile] | None = None,
    ) -> None:
        self._profiles = _DEFAULT_PROFILES if profiles is None else dict(profiles)
        self.coverage_gap = coverage_gap

    # -- gears ------------------------------------------------------------

    def profile(self, gear: ExplorationGear) -> GearProfile:
        return self._profiles[gear]

    def gears(self) -> tuple[ExplorationGear, ...]:
        return tuple(ExplorationGear)

    # -- channel policies ------------------------------------------------

    def _marginal(self, posterior, oid: str):
        if posterior is None:
            return None
        try:
            return posterior.marginal(oid)
        except Exception:
            return None

    def channel_scores(
        self,
        channel: str,
        object_ids: Sequence[str],
        posterior,
        eta: float,
    ) -> dict[str, float]:
        """Raw per-object scores for one within-space channel (SOL/BND/COV).

        OPEN has no in-space scores (it lives in the language track).
        """
        if channel == OPEN:
            return {}
        scores: dict[str, float] = {}
        for oid in object_ids:
            m = self._marginal(posterior, oid)
            p = (m.solution_probability if m and m.solution_probability is not None else 0.0) or 0.0
            u = float(m.std if m and m.std is not None else 1.0)
            if channel == SOL:
                s = float(p)
            elif channel == BND:
                prof = self._default_tau()
                s = u * math.exp(-abs(float(p) - float(eta)) / max(prof, 1e-9))
            else:  # COV
                gap = self.coverage_gap(oid) if self.coverage_gap is not None else float(u)
                s = float(gap) + self.beta_c() * float(u)
            scores[oid] = s
        return scores

    def _default_tau(self) -> float:
        return self._profiles[ExplorationGear.MEDIUM].tau_b

    def beta_c(self) -> float:
        return self._profiles[ExplorationGear.MEDIUM].beta_c

    def normalized(
        self,
        channel: str,
        object_ids: Sequence[str],
        posterior,
        eta: float,
        *,
        mask: set[str] | None = None,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Masked-softmax normalized policy for one channel (pi_t^j).

        Previously-evaluated / invalid / forbidden states are masked before
        selection.  Returns a probability per candidate summing to 1 over the
        allowed set (empty -> {}).
        """
        mask = mask or set()
        scores = self.channel_scores(channel, object_ids, posterior, eta)
        allowed = {oid: s for oid, s in scores.items() if oid not in mask}
        if not allowed:
            return {}
        max_s = max(allowed.values())
        exps = {oid: math.exp((s - max_s) / max(temperature, 1e-9)) for oid, s in allowed.items()}
        z = sum(exps.values()) or 1.0
        return {oid: v / z for oid, v in exps.items()}

    def rank(
        self,
        channel: str,
        object_ids: Sequence[str],
        posterior,
        eta: float,
        *,
        mask: set[str] | None = None,
    ) -> list[str]:
        """Deterministic descending ranking by normalized policy probability."""
        pol = self.normalized(channel, object_ids, posterior, eta, mask=mask)
        return sorted(pol, key=lambda o: pol[o], reverse=True)

    def select_channel(self, rng, gear: ExplorationGear) -> str:
        """Sample a channel from the gear's categorical distribution."""
        prof = self.profile(gear)
        w = prof.weights
        r = rng.random()
        acc = 0.0
        for ch, wi in zip(CHANNELS, w):
            acc += wi
            if r <= acc or wi >= 1.0:
                return ch
        return SOL

    # -- batched allocation ----------------------------------------------

    def batch_plan(
        self,
        q: int,
        gear: ExplorationGear,
        residual: Sequence[float] | None = None,
    ) -> tuple[dict[str, int], tuple[float, float, float, float]]:
        """Allocate integer per-channel counts for a batch of ``q``.

        Uses balanced (largest-remainder) rounding on ``q * w_g`` plus any
        carried rounding residual, so small batches still converge to the
        long-run proportions across rounds.  Returns (counts, new_residual).
        """
        w = self.profile(gear).weights
        res = [0.0, 0.0, 0.0, 0.0] if residual is None else list(residual)
        target = [q * wi + ri for wi, ri in zip(w, res)]
        counts = [int(math.floor(t)) for t in target]
        leftover = q - sum(counts)
        if leftover > 0:
            fracs = [(target[i] - counts[i], i) for i in range(4)]
            fracs.sort(key=lambda t: (-t[0], t[1]))  # tie-break by channel order
            for _ in range(leftover):
                _, i = fracs[_ % len(fracs)]
                counts[i] += 1
        new_residual = tuple(target[i] - counts[i] for i in range(4))
        out = dict(zip(CHANNELS, counts))
        return out, new_residual

    # -- optional recommendation (implementation aid, non-overriding) ------

    def recommend(self, meta: Mapping[str, Any] | None = None) -> ExplorationGear:
        """Suggest a gear from lightweight state signals (never overrides the
        user-selected gear).  v0.1 §Control authority."""
        meta = meta or {}
        if meta.get("pure_exploit"):
            return ExplorationGear.OFF
        if meta.get("representation_inadequate"):
            return ExplorationGear.ULTRA
        if meta.get("coverage_stalled"):
            return ExplorationGear.HIGH
        if meta.get("consolidate_basin"):
            return ExplorationGear.LOW
        return ExplorationGear.MEDIUM
