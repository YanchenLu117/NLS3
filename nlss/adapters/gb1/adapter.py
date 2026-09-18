"""GB1Adapter — GB1 protein combinatorics (§10).

Bridge the GB1 directed-evolution benchmark (Wu 2016, eLife) to the frozen V7
TaskAdapter contract.

Grounded object = a 4-site variant of the GB1 domain (a 4-letter amino-acid
combo at the four mutation sites; wild type ``VDGV``).  The empirical scalar
field is the experimentally measured fitness of the variant.  Per §10.2 FROZEN
boundary, the oracle uses ONLY the 149,361 experimentally measured variants
(never the 10,639 imputed).

Geometry (§10.5 FROZEN): ``build_scientific_graph`` is the Hamming-1 mutation
graph — outcome-blind edges between variants at Hamming distance 1.  The value
cut ``gamma`` (fitness threshold) is passed to the backend via
``solution_threshold`` (T3 gamma semantics); ``eta`` (probability cutoff) lives
on ``space_state.metadata["eta"]``.

Non-leakage: ``evaluate`` returns fitness only for queried variants present in
the measured table; unqueried fitness never enters the posterior.
"""

from __future__ import annotations

import re
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
from .data import GB1Data, N_SITES, WILD_TYPE

OBJECT_TYPE = "gb1_variant"

# Standard 20 amino-acid one-letter alphabet (deterministic order).
_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = set(_AA)
_VARIANT_RE = re.compile(r"^\s*([A-Za-z]{4})\s*$")

# Hamming distance for the FROZEN §10.5 mutation graph.
_MUT_DISTANCE = 1


def _parse_variant(text: str) -> str | None:
    m = _VARIANT_RE.match(text)
    if m is None:
        return None
    v = m.group(1).upper()
    if all(c in _AA_SET for c in v):
        return v
    return None


class GB1Adapter(TaskAdapter):
    """GB1 benchmark adapter.

    Parameters
    ----------
    data:
        A :class:`GB1Data`; if ``None`` a shared lazy instance is used.
    q:
        Quantile for the fitness value cut (default 0.95; ``top_pct`` mode).
    criterion:
        Solution-set criterion.  ``top_pct`` (default) is the top-``(1-q)``
        by measured fitness (§10.2-style).  ``better_wt`` is the §10.4 PRIMARY
        better-than-wild-type criterion ``S*_WT = {x : y(x) > y(WT)}`` with
        ``WT = VDGV``: the recovery target is every measured variant strictly
        better than the wild type, which is the empirically richer solution
        set (|S*_WT| = 3,643, 3 connected components on the full table).
    """

    task_id = "gb1"
    object_type = OBJECT_TYPE

    def __init__(
        self,
        data: GB1Data | None = None,
        *,
        q: float = 0.95,
        criterion: str = "top_pct",
    ) -> None:
        self.data = data if data is not None else GB1Data()
        self.q = q
        if criterion not in ("top_pct", "better_wt"):
            raise ValueError(f"GB1 criterion must be top_pct or better_wt, got {criterion!r}")
        self.criterion = criterion
        self._gamma_cache: float | None = None
        self._wt_fitness_cache: float | None = None

    # -- features (outcome-blind one-hot over the 4 sites) ----------------

    def feature_vectors(self, objects: tuple[GroundedObject, ...]) -> dict[str, np.ndarray]:
        """Outcome-blind 20x4 amino-acid one-hot per variant."""
        out: dict[str, np.ndarray] = {}
        for o in objects:
            v = str(o.payload.get("variant", ""))
            if len(v) != N_SITES:
                continue
            vec = np.zeros(N_SITES * len(_AA), dtype=np.float64)
            for i, ch in enumerate(v):
                if ch in _AA_SET:
                    vec[i * len(_AA) + _AA.index(ch)] = 1.0
            out[o.object_id] = vec
        return out

    # -- grounding: compile -> verify -> canonicalize ---------------------

    async def compile(self, hypothesis: LanguageHypothesis, llm) -> RawScientificObject:
        v = _parse_variant(hypothesis.text)
        if v is None:
            raise GroundingError(
                f"cannot parse GB1 variant from hypothesis {hypothesis.hypothesis_id!r}: "
                f"{hypothesis.text!r} (expected a 4-letter amino-acid combo)"
            )
        return RawScientificObject(
            task_id=self.task_id,
            object_type=self.object_type,
            payload={"variant": v},
            source_hypothesis_id=hypothesis.hypothesis_id,
            raw_text=hypothesis.text,
        )

    def verify(self, raw_object: RawScientificObject) -> VerificationResult:
        v = str(raw_object.payload.get("variant", ""))
        if len(v) != N_SITES or not all(c in _AA_SET for c in v):
            return VerificationResult(
                valid=False, errors=(f"variant must be {N_SITES} standard amino acids",)
            )
        return VerificationResult(valid=True)

    def canonicalize(self, raw_object: RawScientificObject) -> GroundedObject:
        v = str(raw_object.payload["variant"])
        canonical_form = f"{OBJECT_TYPE}|{v}"
        return GroundedObject(
            object_id=stable_object_id(self.task_id, canonical_form),
            task_id=self.task_id,
            object_type=self.object_type,
            canonical_form=canonical_form,
            payload={"variant": v},
            display_text=v,
            source_hypothesis_ids=(raw_object.source_hypothesis_id,),
            metadata={"benchmark": "gb1", "wild_type": WILD_TYPE},
        )

    # -- graphs -----------------------------------------------------------

    def build_scientific_graph(self, objects: tuple[GroundedObject, ...]) -> ScientificGraph:
        ids = [o.object_id for o in objects]
        variants = [str(o.payload.get("variant", "")) for o in objects]
        edges: list[ScientificEdge] = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if _hamming(variants[i], variants[j]) == _MUT_DISTANCE:
                    edges.append(
                        ScientificEdge(
                            edge_id=stable_edge_id(ids[i], ids[j], "gb1_hamming1"),
                            source_id=ids[i],
                            target_id=ids[j],
                            weight=1.0,
                            distance=1.0,
                            relation_type="gb1_hamming1",
                            metadata={"outcome_blind": True, "hamming": _MUT_DISTANCE},
                        )
                    )
        return ScientificGraph(node_ids=tuple(ids), edges=tuple(edges),
                               metadata={"domain": "gb1", "outcome_blind": True,
                                         "geometry": "hamming-1"})

    def build_revision_graph(self, objects: tuple[GroundedObject, ...]) -> RevisionGraph:
        ids = [o.object_id for o in objects]
        variants = [str(o.payload.get("variant", "")) for o in objects]
        edges: list[RevisionEdge] = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if _hamming(variants[i], variants[j]) == _MUT_DISTANCE:
                    edges.append(
                        RevisionEdge(
                            edge_id=stable_edge_id(ids[i], ids[j], "gb1_point_mutation"),
                            source_id=ids[i],
                            target_id=ids[j],
                            action_type="point_mutation",
                            action_description="single-site amino-acid mutation",
                            metadata={"outcome_blind": True, "hamming": _MUT_DISTANCE},
                        )
                    )
        return RevisionGraph(node_ids=tuple(ids), edges=tuple(edges),
                             metadata={"domain": "gb1", "outcome_blind": True})

    # -- evaluation -------------------------------------------------------

    async def evaluate(self, obj: GroundedObject) -> Observation:
        variant = str(obj.payload.get("variant", ""))
        f = self.data.fitness_of(variant)
        if f is None:
            raise ValueError(
                f"variant {variant!r} not in the measured GB1 set; rejected without reveal"
            )
        return Observation(
            object_id=obj.object_id,
            value=float(f),
            round_index=0,
            evaluator="gb1_measured_fitness",
            metadata={"in_measured_table": True},
        )

    def gamma(self) -> float:
        """The active value cut for the configured solution criterion.

        * ``top_pct``:   ``q``-quantile of measured fitness (default q=0.95).
        * ``better_wt``: ``y(WT)`` — every variant strictly above the wild-type
          fitness is in S*_WT (§10.4 PRIMARY).  WT is never itself in S*_WT.
        """
        if self.criterion == "better_wt":
            if self._wt_fitness_cache is None:
                wt = self.data.fitness_of(WILD_TYPE)
                if wt is None:
                    raise ValueError(f"GB1 wild type {WILD_TYPE!r} not in the measured table")
                self._wt_fitness_cache = float(wt)
            return self._wt_fitness_cache
        if self._gamma_cache is None:
            self._gamma_cache = self.data.gamma(self.q)
        return self._gamma_cache

    def wild_type_fitness(self) -> float:
        """Fitness of the wild type ``VDGV`` (the §10.4 reference)."""
        return self.gamma() if self.criterion == "better_wt" else float(self.data.fitness_of(WILD_TYPE))

    def solution_set(self) -> tuple[str, ...]:
        """The variants in the active S* by the configured criterion, sorted
        by descending fitness (stable, reproducible arm).

        * ``top_pct``:   {y(x) >= gamma}.
        * ``better_wt``: {y(x) > y(WT)} (strictly above the wild type, §10.4).
        """
        g = self.gamma()
        if self.criterion == "better_wt":
            recs = [v for v, f in self.data.fitness.items() if f > g]
        else:
            recs = [v for v, f in self.data.fitness.items() if f >= g]
        recs.sort(key=lambda v: -self.data.fitness[v])
        return tuple(recs)

    def solution_threshold(self) -> float:
        """Alias of :meth:`gamma` (value cut passed to the recovery backend)."""
        return self.gamma()

    def criterion_summary(self) -> dict[str, Any]:
        """Verbatim description of the active solution criterion."""
        return {
            "benchmark": "gb1",
            "criterion": self.criterion,
            "q": self.q if self.criterion == "top_pct" else None,
            "wild_type": WILD_TYPE,
            "gamma": self.gamma(),
        }

    def solution_diagnostics(
        self, *, with_components: bool = True
    ) -> dict[str, Any]:
        """Diagnostic block for the active criterion: gamma, |S*|, and the
        connected-component structure of the induced Hamming-1 graph over S*.

        NOTE: component structure is only computed over the SOLUTION SET
        (|S*_WT| = 3,643 or |top-5%| = 7,469), never over the full 149,361
        population — the model loop always builds its graph on the queried
        subset (≤ budget of ~480) per §10.5, so this diagnostic stays O(n^2)
        on the much smaller S* for reporting only.
        """
        sol = self.solution_set()
        out: dict[str, Any] = {
            "criterion": self.criterion,
            "wild_type": WILD_TYPE,
            "gamma": self.gamma(),
            "n_total": self.data.effective_size(),
            "n_solution": len(sol),
            "solution_fraction": (len(sol) / self.data.effective_size()) if self.data.effective_size() else 0.0,
        }
        if with_components:
            out["components"] = _component_structure(sol)
        return out

    # -- rendering / FrontierSemantics ------------------------------------

    def render_object(self, obj: GroundedObject) -> str:
        return obj.display_text

    def descriptor(self, obj: GroundedObject) -> dict[str, Any]:
        v = str(obj.payload.get("variant", ""))
        return {
            "site0": v[0] if len(v) > 0 else "",
            "site1": v[1] if len(v) > 1 else "",
            "site2": v[2] if len(v) > 2 else "",
            "site3": v[3] if len(v) > 3 else "",
            "n_sites": N_SITES,
            "wild_type": WILD_TYPE,
        }

    def attribute_domains(self) -> dict[str, tuple[Any, ...]]:
        """Per-site amino-acid domains (PUBLIC mutation-space grammar)."""
        aa = tuple(_AA)
        return {
            "site0": aa,
            "site1": aa,
            "site2": aa,
            "site3": aa,
            "n_sites": (N_SITES,),
            "wild_type": (WILD_TYPE,),
        }

    def cell_feasible(self, descriptor: Mapping[str, Any]) -> bool:
        return all(
            descriptor.get(f"site{i}") in _AA_SET for i in range(N_SITES)
        )

    def revision_hints(
        self, from_descriptor: Mapping[str, Any], to_descriptor: Mapping[str, Any]
    ) -> tuple[str, ...]:
        needs = sum(
            1
            for i in range(N_SITES)
            if from_descriptor.get(f"site{i}") != to_descriptor.get(f"site{i}")
        )
        return ("point_mutation",) * min(needs, 4) or ("point_mutation",)

    def target_instruction(self, descriptor: Mapping[str, Any]) -> str:
        v = "".join(str(descriptor.get(f"site{i}", "")) for i in range(N_SITES))
        base = "Propose a GB1 variant "
        if v:
            base += f"near {v} "
        return base.strip() + "by single amino-acid mutations, distinct from anchors, to maximize fitness."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hamming(a: str, b: str) -> int:
    return sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i]) + abs(len(a) - len(b))


def _component_structure(solution: tuple[str, ...]) -> dict[str, Any]:
    """Connected components of the Hamming-1 mutation graph induced on the
    solution set S* (§10.5 geometry).  Outcome-blind and computed only on S*
    (never the full population). Returns ``{n_nodes, n_components,
    largest_component, component_sizes}``."""
    n = len(solution)
    if n == 0:
        return {"n_nodes": 0, "n_components": 0, "largest_component": 0, "component_sizes": []}
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        a = solution[i]
        for j in range(i + 1, n):
            if _hamming(a, solution[j]) == _MUT_DISTANCE:
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
