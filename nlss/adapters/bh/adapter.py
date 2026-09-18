"""BHAdapter — Buchwald-Hartwig high-throughput experimentation (§9).

Bridge the BH HTE benchmark (doylelab/rxnpredict) to the frozen V7
TaskAdapter contract.

Grounded object = a reaction candidate ``x = (aryl_halide, ligand, base,
additive)`` (a four-component tuple).  The empirical scalar field is the
measured yield ``y(x) in [0, 100]``; the frozen solution set is
``S* = { x : y(x) >= q_0.95(y) }`` (top-5% yield, §9.4 FROZEN).  The value cut
``gamma = q_0.95(y)`` is passed to the recovery backend via the model's
``solution_threshold`` (T3 gamma semantics) — a value-space cut, never a
probability threshold (eta lives on ``space_state.metadata["eta"]``).

Non-leakage discipline: :meth:`evaluate` returns the yield for exactly the
queried candidate; it never bulk-exposes the full table, and unqueried yields
never enter the posterior (built only from queried observations).

Outcome-blind geometry: no rdkit in the nlss-core env, so :meth:`feature_vectors`
uses a deterministic component one-hot and :meth:`build_scientific_graph` uses
outcome-blind one-hot cosine similarity.  ``build_revision_graph`` = one-factor
substitution (change a single component), matching §9.7 "one_factor_replacement".
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...core.errors import GroundingError
from ...core.ids import stable_edge_id, stable_object_id
from ...core.interfaces import TaskAdapter
from ...core.types import (
    GroundedObject,
    LanguageHypothesis,
    Observation,
    RawScientificObject,
    RevisionEdge,
    RevisionGraph,
    ScientificEdge,
    ScientificGraph,
    VerificationResult,
)
from .data import BHData, Candidate

OBJECT_TYPE = "bh_reaction"
_DEFAULT_Q = 0.95
_COMPONENT_KEYS = ("aryl_halide", "ligand", "base", "additive")

# A canonical candidate renders as ``aryl_halide|ligand|base|additive``.
# Component names are SMILES-free identifiers; ``|`` / ``,`` are unambiguous
# separators (see ``_parse_candidate``).


def _parse_candidate(text: str) -> Candidate | None:
    """Parse ``a|l|b|d`` (additive optional, may be empty) -> 4-tuple.

    Component names in the BH data may themselves contain commas (e.g. the
    additive ``5-Phenyl-1,2,4-oxadiazole``), so the separator MUST be the pipe
    ``|`` only — comma is NOT a safe separator here.
    """
    parts = [p.strip() for p in text.split("|")]
    if len(parts) < 3:
        return None
    a, l, b = parts[0], parts[1], parts[2]
    d = parts[3] if len(parts) > 3 else ""
    if not a or not l or not b:
        return None
    return (a, l, b, d)


class BHAdapter(TaskAdapter):
    """BH benchmark adapter.

    Parameters
    ----------
    data:
        A :class:`BHData`; if ``None`` a shared lazy instance is used.
    q:
        Quantile for the frozen top-``(1-q)`` solution threshold (default 0.95
        => top-5% yield, §9.4).
    """

    task_id = "bh"
    object_type = OBJECT_TYPE

    def __init__(
        self,
        data: BHData | None = None,
        *,
        q: float = _DEFAULT_Q,
        criterion: str = "top_pct",
        top_frac: float | None = None,
        fixed_yield: float | None = None,
    ) -> None:
        """Construct the BH adapter with a configurable solution-set criterion.

        The frozen *primary* solution set is ``S* = {x : y(x) >= q_0.95(y)}``
        (§9.4).  §9.4 also specifies an OPTIONAL *secondary significance arm* to
        guard against the power risk that the top-5% set (|S*| = 230, nearly the
        240-budget) leaves too little headroom to separate methods.  The arm is
        selected by ``criterion``:

        * ``top_pct``   — S* = top ``top_frac`` fraction of the *measured*
          population by yield (default ``top_frac = 1 - q``, so ``q=0.95``
          reproduces the frozen top-5% primary; ``top_frac=0.10`` gives the
          top-10% secondary arm).  ``gamma`` is the ``(1 - top_frac)``
          quantile of measured yields.
        * ``fixed``     — S* = {x : y(x) >= ``fixed_yield``} (absolute yield
          cut, e.g. ``fixed_yield=80`` per §9.4 "yield>=80 if enough positives").
          ``gamma`` is exactly ``fixed_yield``.

        The active arm is reported verbatim in :meth:`criterion_summary` and in
        every thread-facing diagnostic so the comparison never mixes arms.

        Parameters
        ----------
        data:
            A :class:`BHData`; if ``None`` a shared lazy instance is used.
        q:
            Quantile for the top-``(1-q)`` fraction (legacy; only used in
            ``top_pct`` mode and only when ``top_frac`` is not given).  Default
            0.95 => top-5% primary (§9.4).
        criterion:
            ``top_pct`` (default) or ``fixed``.
        top_frac:
            Absolute top fraction for ``top_pct`` mode (e.g. 0.10 for §9.4
            secondary arm).  If ``None`` falls back to ``1 - q``.
        fixed_yield:
            Absolute yield threshold (0-100) for ``fixed`` mode.
        """
        self.data = data if data is not None else BHData()
        if criterion not in ("top_pct", "fixed"):
            raise ValueError(f"BH criterion must be top_pct or fixed, got {criterion!r}")
        self.criterion = criterion
        self._top_frac = float(top_frac) if top_frac is not None else None
        self.q = float(q)
        self.fixed_yield = float(fixed_yield) if fixed_yield is not None else None
        # Precompute vocab + one-hot index once (cheap, deterministic).
        self._vocab_cache: dict[str, tuple[str, ...]] | None = None
        self._gamma_cache: float | None = None

    def _active_top_frac(self) -> float:
        """Effective top fraction for ``top_pct`` mode (top_frac or 1-q)."""
        if self._top_frac is not None:
            return self._top_frac
        return 1.0 - self.q

    # -- vocabulary / features -------------------------------------------

    def _vocab(self) -> dict[str, tuple[str, ...]]:
        if self._vocab_cache is None:
            self._vocab_cache = self.data.component_vocab()
        return self._vocab_cache

    def _siso(self, cand: Candidate) -> list[float]:
        """Single component one-hot over the measured vocab (outcome-blind)."""
        vocab = self._vocab()
        vec: list[float] = []
        for i, key in enumerate(_COMPONENT_KEYS):
            domain = vocab[key]
            hot = [0.0] * len(domain)
            val = cand[i]
            if val in domain:
                hot[domain.index(val)] = 1.0
            else:
                # unknown component => all-zeros for that block (still fixed-dim)
                pass
            vec.extend(hot)
        return vec

    def feature_vectors(self, objects: tuple[GroundedObject, ...]) -> dict[str, np.ndarray]:
        """Outcome-blind component one-hot per object (no rdkit available)."""
        out: dict[str, np.ndarray] = {}
        for o in objects:
            cand = _candidate_of(o)
            if cand is None:
                continue
            out[o.object_id] = np.asarray(self._siso(cand), dtype=np.float64)
        return out

    # -- grounding: compile -> verify -> canonicalize ---------------------

    async def compile(self, hypothesis: LanguageHypothesis, llm) -> RawScientificObject:
        cand = _parse_candidate(hypothesis.text)
        if cand is None:
            raise GroundingError(
                f"cannot parse BH candidate from hypothesis {hypothesis.hypothesis_id!r}: "
                f"{hypothesis.text!r} (expected aryl_halide|ligand|base|additive)"
            )
        return RawScientificObject(
            task_id=self.task_id,
            object_type=self.object_type,
            payload={"candidate": cand},
            source_hypothesis_id=hypothesis.hypothesis_id,
            raw_text=hypothesis.text,
        )

    def verify(self, raw_object: RawScientificObject) -> VerificationResult:
        cand = raw_object.payload.get("candidate")
        if not isinstance(cand, tuple) or len(cand) != 4:
            return VerificationResult(valid=False, errors=("candidate must be a 4-tuple",))
        if not all(str(c).strip() for c in cand[:3]):
            return VerificationResult(valid=False, errors=("aryl_halide/ligand/base required",))
        return VerificationResult(valid=True)

    def canonicalize(self, raw_object: RawScientificObject) -> GroundedObject:
        cand = tuple(str(c) for c in raw_object.payload["candidate"])
        canonical_form = _canonical_form(cand)
        return GroundedObject(
            object_id=stable_object_id(self.task_id, canonical_form),
            task_id=self.task_id,
            object_type=self.object_type,
            canonical_form=canonical_form,
            payload={"candidate": cand},
            display_text=_canonical_form(cand),
            source_hypothesis_ids=(raw_object.source_hypothesis_id,),
            metadata={"benchmark": "bh", "dataset": "rxnpredict_data_table"},
        )

    # -- graphs -----------------------------------------------------------

    def build_scientific_graph(self, objects: tuple[GroundedObject, ...]) -> ScientificGraph:
        ids = [o.object_id for o in objects]
        n = len(ids)
        edges: list[ScientificEdge] = []
        if n <= 1:
            return ScientificGraph(node_ids=tuple(ids), edges=tuple(edges),
                                   metadata={"domain": "bh", "outcome_blind": True})
        cands = [_candidate_of(o) for o in objects]
        feats = np.asarray([self._siso(c) for c in cands], dtype=np.float64)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
        x = feats / norms
        sim = x @ x.T
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sim[i, j])
                if s > 0.0:
                    edges.append(
                        ScientificEdge(
                            edge_id=stable_edge_id(ids[i], ids[j], "bh_component_similarity"),
                            source_id=ids[i],
                            target_id=ids[j],
                            weight=s,
                            distance=1.0 - s,
                            relation_type="bh_component_similarity",
                            metadata={"outcome_blind": True},
                        )
                    )
        return ScientificGraph(node_ids=tuple(ids), edges=tuple(edges),
                               metadata={"domain": "bh", "outcome_blind": True})

    def build_revision_graph(self, objects: tuple[GroundedObject, ...]) -> RevisionGraph:
        ids = [o.object_id for o in objects]
        cands = [_candidate_of(o) for o in objects]
        edges: list[RevisionEdge] = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                d = _hamming(cands[i], cands[j])
                if d == 1:
                    edges.append(
                        RevisionEdge(
                            edge_id=stable_edge_id(ids[i], ids[j], "bh_one_factor_substitution"),
                            source_id=ids[i],
                            target_id=ids[j],
                            action_type="one_factor_substitution",
                            action_description="scientific one-factor substitution",
                            metadata={"outcome_blind": True, "hamming": 1},
                        )
                    )
        return RevisionGraph(node_ids=tuple(ids), edges=tuple(edges),
                             metadata={"domain": "bh", "outcome_blind": True})

    # -- evaluation -------------------------------------------------------

    async def evaluate(self, obj: GroundedObject) -> Observation:
        cand = _candidate_of(obj)
        if cand is None:
            raise ValueError(f"object {obj.object_id} has no evaluable BH candidate")
        y = self.data.yield_of(cand)
        if y is None:
            # Candidate absent from the measured table: reject WITHOUT revealing
            # any yield (only queried yields are ever returned).
            raise ValueError(
                f"candidate {cand!r} not in the measured BH table; rejected without reveal"
            )
        return Observation(
            object_id=obj.object_id,
            value=float(y),
            round_index=0,
            evaluator="bh_measured_yield",
            metadata={"in_measured_table": True},
        )

    def gamma(self) -> float:
        """The active value cut gamma for the *configured* solution arm.

        * ``top_pct``: gamma = ``(1 - top_frac)`` quantile of measured yields
          (the frozen primary, ``q=0.95`` => top-5%, is the default).
        * ``fixed``:   gamma = ``fixed_yield`` (absolute yield cut).
        """
        if self._gamma_cache is None:
            if self.criterion == "fixed":
                if self.fixed_yield is None:
                    raise ValueError("BH criterion=fixed requires fixed_yield")
                self._gamma_cache = self.fixed_yield
            else:
                self._gamma_cache = self.data.gamma(1.0 - self._active_top_frac())
        return self._gamma_cache

    def solution_threshold(self) -> float:
        """Alias of :meth:`gamma` (value cut passed to the recovery backend)."""
        return self.gamma()

    def criterion_summary(self) -> dict[str, Any]:
        """Verbatim description of the active solution arm (never mixes arms)."""
        return {
            "benchmark": "bh",
            "criterion": self.criterion,
            "top_frac": self._active_top_frac() if self.criterion == "top_pct" else None,
            "fixed_yield": self.fixed_yield if self.criterion == "fixed" else None,
            "gamma": self.gamma(),
        }

    def solution_set(self) -> tuple[Candidate, ...]:
        """The measured candidates in the active S* (by the configured arm).

        Returns tuples sorted by descending yield for a stable, reproducible
        arm; ``len(S*)`` is the arm size (diagnostic ``|S*|``).
        """
        g = self.gamma()
        recs = [c for c, y in self.data.records if y >= g]
        recs.sort(key=lambda c: -self.data.yield_of(c))
        return tuple(recs)

    def solution_diagnostics(self, *, with_components: bool = True) -> dict[str, Any]:
        """Diagnostic block for the active arm: gamma, |S*|, connected
        component structure of the induced one-hot similarity graph over S*.

        Every arm exposes the same diagnostic keys so the multi-arm comparison
        (primary top-5% vs secondary top-10% vs fixed threshold) is aligned.
        """
        sol = self.solution_set()
        g = self.gamma()
        out: dict[str, Any] = {
            "criterion": self.criterion,
            "top_frac": self._active_top_frac() if self.criterion == "top_pct" else None,
            "fixed_yield": self.fixed_yield if self.criterion == "fixed" else None,
            "gamma": g,
            "n_total": self.data.effective_size(),
            "n_solution": len(sol),
            "solution_fraction": (len(sol) / self.data.effective_size()) if self.data.effective_size() else 0.0,
        }
        if with_components:
            out["components"] = _component_structure(self, sol)
        return out

    def top_pct(self) -> float:
        """Effective top fraction of the population for the active arm
        (for the fixed arm this is the realized fraction above gamma)."""
        if self.criterion == "fixed":
            return len(self.solution_set()) / self.data.effective_size() if self.data.effective_size() else 0.0
        return self._active_top_frac()

    # -- rendering / FrontierSemantics ------------------------------------

    def render_object(self, obj: GroundedObject) -> str:
        return obj.display_text

    def descriptor(self, obj: GroundedObject) -> dict[str, Any]:
        cand = _candidate_of(obj)
        vocab = self._vocab()
        return {
            "aryl_halide": cand[0],
            "ligand": cand[1],
            "base": cand[2],
            "additive": cand[3],
            "n_components": len(_COMPONENT_KEYS),
        }

    def target_instruction(self, descriptor: Mapping[str, Any]) -> str:
        a = descriptor.get("aryl_halide")
        l = descriptor.get("ligand")
        b = descriptor.get("base")
        d = descriptor.get("additive")
        base = "Propose a BH reaction candidate "
        if a:
            base += f"with aryl halide {a} "
        if l:
            base += f"ligand {l} "
        if b:
            base += f"base {b} "
        if d:
            base += f"additive {d} "
        return base.strip() + ", distinct from all anchors, aiming to maximize yield within budget."

    def semantic_kind(self) -> str:
        return "value_field"

    # -- FrontierSemantics (readout-v2) ------------------------------------

    def attribute_domains(self) -> dict[str, tuple[Any, ...]]:
        """Component vocabularies (PUBLIC grammar product space)."""
        vocab = self._vocab()
        additive = ("",) + vocab["additive"]  # additive is optional (empty == none)
        return {
            "aryl_halide": vocab["aryl_halide"],
            "ligand": vocab["ligand"],
            "base": vocab["base"],
            "additive": additive,
            "n_components": (4,),
        }

    def cell_feasible(self, descriptor: Mapping[str, Any]) -> bool:
        """A cell is feasible iff each present component is in the known vocab."""
        vocab = self._vocab()
        for key in ("aryl_halide", "ligand", "base"):
            v = descriptor.get(key)
            if v not in vocab[key]:
                return False
        add = descriptor.get("additive", "")
        if add not in (("",) + vocab["additive"]):
            return False
        return True

    def revision_hints(
        self, from_descriptor: Mapping[str, Any], to_descriptor: Mapping[str, Any]
    ) -> tuple[str, ...]:
        hints: list[str] = []
        if to_descriptor.get("aryl_halide") != from_descriptor.get("aryl_halide"):
            hints.append("one_factor_substitution")
        if to_descriptor.get("ligand") != from_descriptor.get("ligand"):
            hints.append("one_factor_substitution")
        if to_descriptor.get("base") != from_descriptor.get("base"):
            hints.append("one_factor_substitution")
        if to_descriptor.get("additive") != from_descriptor.get("additive"):
            hints.append("one_factor_substitution")
        return tuple(dict.fromkeys(hints)) or ("one_factor_substitution",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_of(obj: GroundedObject) -> Candidate | None:
    c = obj.payload.get("candidate")
    if isinstance(c, tuple) and len(c) == 4:
        return (str(c[0]), str(c[1]), str(c[2]), str(c[3]))
    return None


def _canonical_form(cand: Candidate) -> str:
    return f"{OBJECT_TYPE}|{'|'.join(cand)}"


def _hamming(a: Candidate, b: Candidate) -> int:
    return sum(1 for i in range(4) if a[i] != b[i])


def _component_structure(adapter: "BHAdapter", solution: tuple[Candidate, ...]) -> dict[str, Any]:
    """Connected components of the outcome-blind one-hot similarity graph
    induced on the solution set S*, thresholded at any positive similarity.

    Uses the same geometry as :meth:`build_scientific_graph` (component
    one-hot cosine similarity) so the diagnostic matches the graph the
    recovery backend actually sees.  Returns ``{n_nodes, n_components,
    largest_component, component_sizes}``.
    """
    n = len(solution)
    if n == 0:
        return {"n_nodes": 0, "n_components": 0, "largest_component": 0, "component_sizes": []}
    feats = np.asarray([adapter._siso(c) for c in solution], dtype=np.float64)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    feats = feats / norms
    sim = feats @ feats.T
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > 0.0:
                adj[i].add(j)
                adj[j].add(i)
    seen = [False] * n
    sizes: list[int] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        cnt = 0
        while stack:
            u = stack.pop()
            cnt += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        sizes.append(cnt)
    sizes.sort(reverse=True)
    return {
        "n_nodes": n,
        "n_components": len(sizes),
        "largest_component": sizes[0] if sizes else 0,
        "component_sizes": sizes,
    }
