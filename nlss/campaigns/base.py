"""Campaign engine — reusable finite-oracle experiment runner (NLSS V7).

Moves the BH feasibility logic from a one-off imperative script into a
first-class, reproducible campaign engine:

  * a finite graded-oracle abstraction (candidate pool + hidden gold values +
    solution membership) that the **evaluator only** may read;
  * a ``PairedSeedScheduler`` so every arm sees the same initial set per seed;
  * an ``AcquisitionPolicy`` seam so baseline arms (random / greedy / history /
    scalar / DKL-BO) and the NLSS arm all consume the *same* oracle budget
    through one interface;
  * a §12.3-compliant ``ResourceLedger`` (oracle_calls / llm_calls / tokens /
    wallclock / git_commit / config_hash) emitted per run.

This module is deliberately benchmark-agnostic; concrete instantiations (BH,
GB1) live in sibling modules and only supply the oracle + solution criterion.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence


# ---------------------------------------------------------------------------
# Finite graded oracle (evaluator-only)
# ---------------------------------------------------------------------------


class FiniteOracle(Protocol):
    """A finite candidate universe with hidden gold values.

    The recovery *method* may only call ``evaluate(candidate)`` within its
    budget; methods must never see ``gold_of`` / ``solution_membership`` /
    ``gold_values`` (evaluator-only).
    """

    @property
    def candidates(self) -> Sequence[Any]: ...

    def evaluate(self, candidate: Any) -> float:
        """Return the revealed outcome for a candidate (within budget)."""

    def gold_of(self, candidate: Any) -> float: ...  # evaluator-only

    def solution_membership(self, candidate: Any) -> bool: ...  # evaluator-only

    @property
    def gold_values(self) -> Sequence[float]: ...  # evaluator-only


@dataclass
class OracleBudgetExceededError(RuntimeError):
    message: str = "oracle budget exceeded"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """Shared campaign parameters.

    ``n_initial`` + ``rounds`` x ``batch_size`` = total oracle budget (§9.6).
    ``checkpoints`` are cumulative budgets at which recovery/optimization
    metrics are evaluated over the *unqueried* candidates.
    """

    n_initial: int = 40
    batch_size: int = 20
    rounds: int = 10
    checkpoints: tuple[int, ...] = (40, 80, 120, 160, 200, 240)

    @property
    def total_budget(self) -> int:
        return self.n_initial + self.rounds * self.batch_size


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One experiment run's identity + resource metadata (§12.3 log)."""

    benchmark: str
    method: str
    seed: int
    config_hash: str
    git_commit: str = ""

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "method": self.method,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
        }


# ---------------------------------------------------------------------------
# ResourceLedger — §12.3 canonical run log
# ---------------------------------------------------------------------------


class ResourceLedger:
    """Tracks the four §12 fairness budgets and emits a canonical JSON log."""

    def __init__(self, run: RunConfig) -> None:
        self.run = run
        self.oracle_calls = 0
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.surrogate_fit_calls = 0
        self.surrogate_train_seconds = 0.0
        self.started = time.time()
        self._extra: dict[str, Any] = {}

    def bump_oracle(self, n: int = 1) -> None:
        self.oracle_calls += n

    def bump_llm(self, tok_in: int = 0, tok_out: int = 0, calls: int = 0) -> None:
        self.llm_calls += calls
        self.tokens_in += tok_in
        self.tokens_out += tok_out

    def bump_surrogate_fit(self, seconds: float) -> None:
        self.surrogate_fit_calls += 1
        self.surrogate_train_seconds += seconds

    def set(self, **kw: Any) -> None:
        self._extra.update(kw)

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def wall_clock_seconds(self) -> float:
        return time.time() - self.started

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.run.to_log_dict())
        d.update(
            {
                "oracle_calls_cumulative": self.oracle_calls,
                "llm_calls_cumulative": self.llm_calls,
                "tokens_in_cumulative": self.tokens_in,
                "tokens_out_cumulative": self.tokens_out,
                "tokens_total_cumulative": self.tokens_total,
                "surrogate_fit_calls": self.surrogate_fit_calls,
                "surrogate_train_seconds": round(self.surrogate_train_seconds, 3),
                "wall_clock_seconds": round(self.wall_clock_seconds, 3),
                "checkpoint_metrics": self._extra.get("checkpoint_metrics"),
                "final_metrics": self._extra.get("final_metrics"),
            }
        )
        return d

    def dump_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Config hashing / git capture
# ---------------------------------------------------------------------------


def hash_config(config: Mapping[str, Any]) -> str:
    """Stable JSON-sorted hash of a campaign config (for §12.3 config_hash)."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def git_commit(repo_root: str) -> str:
    """Best-effort current commit SHA (empty string if unavailable)."""
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Acquisition policy seam
# ---------------------------------------------------------------------------


class AcquisitionPolicy(Protocol):
    """An oracle-budgeted acquisition arm.

    ``acquire`` receives the oracle (never the gold), the candidates already
    queried in this episode, and how many new candidates to return.  The caller
    evaluates those candidates and feeds the outcomes back via ``observe`` for
    stateful arms (NLSS / history / scalar).

    Implementations MUST NOT read ``oracle.gold_of`` / ``solution_membership`` /
    ``gold_values``.
    """

    name: str

    def reset(self, seed: int) -> None: ...

    def observe(self, candidate: Any, outcome: float) -> None:
        """Feed an evaluated (candidate -> outcome) back into the arm's state."""

    def acquire(self, oracle: FiniteOracle, queried_cands: set[Any], n: int) -> list[Any]:
        """Return up to ``n`` candidate proposals from the unqueried pool."""


def default_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # src/nlss/campaigns -> repo root = ../../../..
    return os.path.abspath(os.path.join(here, "..", "..", ".."))
