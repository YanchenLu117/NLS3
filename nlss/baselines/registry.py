"""V7 baselines — unified comparator registry (EXPERIMENT_PLAN §5).

Two blocks, selectable by name so the run-driver (controlled/native regimes)
assembles arms without touching adapter internals:

CONTROLED comparators (proposers that pick the next batch from a candidate pool
given history — measured on the SAME recovery/utility metrics as NLSS):
  - a0_history        : A0 — repeats the best/observed history greedily
  - a1_summary        : A1 — token-matched prose summary context (LLM-generated;
                         this factory is a documented arm that the run-driver must
                         feed with a summary; selection itself is greedy helper)
  - a2_scalar_diversity: A2 — scalar target + diversity (uncertainty/novelty mix)
  - a3_gp_lse         : A3 — GP contemporary level-set classification (replaces
                         BALLET; see baselines.gp_lse)
  - a4_full_nlss      : A4 — the full NLSS loop (marker: run-driver runs the loop;
                         factory returns an NLSSModel builder)

SPECIALIST block:
  - gp_bo, dkl_bo, alde, random, greedy_hillclimb

Honesty: the main-matrix arms A0-A3 and the specialist gp_bo/a3 are REAL
deterministic proposers; A4 is a marker for the full NLSS loop (run-driver).
Heavy specialist arms that need external repos (ALDE) or a deep-kernel backend
(DKL-BO) are documented SHIMS that RAISE unless supplied — they are NOT yet
runnable; the main controlled matrix (A0-A4 + gp_bo) IS runnable now.  Living SR/
Spatial-restore note: external AI-scientist systems (S2/S3/S4) are wired at the
adapter boundary (nlss/adapters/aiscientist, baselines/native_arms), not here.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .gp_bo import GPBandit
from .gp_lse import GPLSE

# ---- simple deterministic proposers ------------------------------------


class _Proposer:
    def __init__(self, **kw) -> None:
        self.kw = kw

    def select(self, pool, ids=None, k=1):
        raise NotImplementedError

    def info(self):
        return {"name": self.__class__.__name__, "cfg": dict(self.kw)}


class HistoryProposer(_Proposer):
    """A0: greedily repeat the best historical candidate (mode of history)."""

    def select(self, pool, ids=None, k=1):
        hist = self.kw.get("history", {})
        ordered = sorted(hist, key=lambda o: -float(hist[o]))
        chosen = ordered[:k] if ordered else (list(pool)[:k] if not isinstance(pool, dict) else list(pool)[:k])
        return tuple(chosen), ()
 
    def info(self):
        return {"name": "a0_history", "note": "greedy-repeat of best observed history"}


class SummaryProposer(_Proposer):
    """A1: token-matched prose summary of history under the plan's ``token_cap``
    (same budget as NLSS's serialized state — the memory-compression baseline).

    Deterministic compressor: top observed candidates by value, truncated to the
    token cap; selection replays the top of that summary.  An LLM would produce
    a richer summary, but the token-matched compressor is the core A1 semantics
    (exclude memory-compression advantage, §5 A1).
    """

    def __init__(self, token_cap: int = 1600, **kw) -> None:
        super().__init__(**kw)
        self.token_cap = max(1, int(token_cap))
        self._history: dict[str, float] = {}

    def set_history(self, history) -> "SummaryProposer":
        self._history = {str(k): float(v) for k, v in dict(history).items()}
        return self

    def summarize(self) -> str:
        hist = dict(self.kw.get("history", {}))
        hist.update(self._history)
        parts, used = [], 0
        for o in sorted(hist, key=lambda x: -float(hist[x])):
            line = f"{o}: {float(hist[o]):.3g}"
            w = len(line.split())
            if used + w > self.token_cap:
                break
            parts.append(line)
            used += w
        return " | ".join(parts)

    def select(self, pool, ids=None, k=1):
        hist = dict(self.kw.get("history", {}))
        hist.update(self._history)
        idmap = set(ids) if ids is not None else None
        ordered = [o for o in sorted(hist, key=lambda x: -float(hist[x]))]
        if idmap:
            ordered = [o for o in ordered if o in idmap]
        return tuple(ordered[:k]), ()

    def info(self):
        return {"name": "a1_summary", "token_cap": self.token_cap,
                "note": "token-matched prose summary (deterministic compressor)"}


class ScalarDiversityProposer(_Proposer):
    """A2: real scalar + diversity (QD-style greedy) scorer over the candidate
    pool.  Each pool entry may be a scalar or (features..., scalar).  Score =
    ``alpha*normalize(scalar) + lam*min-distance-to-selected``; greedy top-k.
    ``set_scalars``/``set_features`` let the run-driver inject domain scalars /
    feature vectors per object id."""

    def __init__(self, lam: float = 0.5, alpha: float = 1.0, **kw) -> None:
        super().__init__(**kw)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self._scalars: dict[str, float] = {}
        self._features: dict[str, Sequence] = {}

    def set_scalars(self, scalars) -> "ScalarDiversityProposer":
        self._scalars = {str(k): float(v) for k, v in dict(scalars).items()}
        return self

    def set_features(self, features) -> "ScalarDiversityProposer":
        self._features = {str(k): v for k, v in dict(features).items()}
        return self

    def _scalar_for(self, oid: str, pool, idx: int) -> float:
        if oid in self._scalars:
            return float(self._scalars[oid])
        if pool is not None and idx < len(pool):
            e = pool[idx]
            if isinstance(e, (list, tuple)) and len(e) >= 2 and isinstance(e[-1], (int, float)):
                return float(e[-1])
        return 0.0

    def _dist(self, a: str, b: str) -> float:
        va, vb = self._features.get(a), self._features.get(b)
        if va is not None and vb is not None and len(va) == len(vb):
            return float(np.linalg.norm(np.asarray(va, dtype=float) - np.asarray(vb, dtype=float)))
        # scalar-distance proxy when no feature vectors
        return abs(self._scalars.get(a, 0.0) - self._scalars.get(b, 0.0))

    def select(self, pool, ids=None, k=1):
        ids_list = [str(i) for i in (ids if ids is not None else range(len(pool) if pool else 0))]
        if not ids_list and isinstance(pool, dict):
            ids_list = [str(x) for x in pool]
        scalars = {oid: self._scalar_for(oid, pool, idx) for idx, oid in enumerate(ids_list)}
        chosen: list[str] = []
        for _ in range(max(0, k)):
            best, best_score = None, -1e18
            for oid in ids_list:
                if oid in chosen:
                    continue
                div = min([self._dist(oid, j) for j in chosen], default=0.0) if chosen else 0.0
                score = self.alpha * scalars[oid] + self.lam * div
                if score > best_score:
                    best_score, best = score, oid
            if best is None:
                break
            chosen.append(best)
        return tuple(chosen), ()

    def info(self):
        return {"name": "a2_scalar_diversity", "lam": self.lam,
                "note": "scalar + min-distance diversity (QD-style greedy)"}


class GPProposer(_Proposer):
    def __init__(self, backend: str, **kw) -> None:
        super().__init__(**kw)
        self.backend = backend
        self._gp = GPBandit(**kw) if backend == "gp_bo" else None
        self._lse = GPLSE(threshold=float(kw.get("threshold", 0.5)), beta=kw.get("beta", 2.0)) if backend == "gp_lse" else None

    def update(self, xs, ys):
        if self._gp:
            self._gp.update(xs, ys)
        if self._lse:
            self._lse.update(xs, ys)
        return self

    def select(self, pool, ids=None, k=1):
        if self._gp:
            chosen, scores = self._gp.select(pool, ids=ids, k=k)
            return chosen, scores
        if self._lse:
            dec = self._lse.select(pool, ids=ids, k=k)
            return dec.ids, dec.scores
        raise RuntimeError(self.backend)

    def info(self):
        return {"name": self.backend,
                **({"note": "GP-LSE (Gotovos 2013 variant)"} if self.backend == "gp_lse" else {}),
                **({"note": "GP-UCB"} if self.backend == "gp_bo" else {})}


class _UnavailableProposer(_Proposer):
    def __init__(self, name: str, reason: str) -> None:
        super().__init__()
        self.name = name
        self.reason = reason

    def select(self, pool, ids=None, k=1):
        raise RuntimeError(f"arm '{self.name}' unavailable: {self.reason}")

    def info(self):
        return {"name": self.name, "available": False, "reason": self.reason}


# ---- registry -----------------------------------------------------------

BASELINES: dict[str, Callable[[Mapping[str, Any] | None], _Proposer]] = {}


def register_baseline(name: str, factory) -> None:
    BASELINES[name] = factory


def get_baseline(name: str):
    if name not in BASELINES:
        raise KeyError(f"unknown baseline: {name}")
    return BASELINES[name]


def available_baselines() -> tuple[str, ...]:
    return tuple(sorted(BASELINES))


def _register_builtins() -> None:
    register_baseline("a0_history", lambda cfg=None: HistoryProposer(**(cfg or {})))
    register_baseline("a2_scalar_diversity", lambda cfg=None: ScalarDiversityProposer(**(cfg or {})))
    register_baseline("a3_gp_lse", lambda cfg=None: GPProposer("gp_lse", **(cfg or {})))
    register_baseline("gp_bo", lambda cfg=None: GPProposer("gp_bo", **(cfg or {})))
    register_baseline("random", lambda cfg=None: _UnavailableProposer("random", "run-driver supplies random selection over pool"))
    register_baseline("a1_summary", lambda cfg=None: SummaryProposer(**(cfg or {})))
    register_baseline("a4_full_nlss", lambda cfg=None: _UnavailableProposer("a4_full_nlss", "marker: run-driver runs the full NLSS loop"))
    register_baseline("dkl_bo", lambda cfg=None: _UnavailableProposer("dkl_bo", "requires deep kernel (external dep); documented shim"))
    register_baseline("alde", lambda cfg=None: _UnavailableProposer("alde", "wraps ALDE specialist (external repo); documented shim"))


_register_builtins()


def create_baseline(name: str, cfg: Mapping[str, Any] | None = None) -> _Proposer:
    return get_baseline(name)(dict(cfg) if cfg else None)
