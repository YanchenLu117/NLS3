"""SpaceReadout — the V6.1 formal component (Core).

Governance: two benchmarks (MADE, HypoSpace) exposed the same conceptual
defect — *having* a solution-space state does not equal *having an effective
readout* from that state.  V6.1 reopens the core **only** for Space Readout:

    S_t -> rho(F_t) -> LLM

where F_t is NOT a summary of S_t but a set of **solution-space frontiers**:
explicit, distinct types of space-expansion directions worth trying next.
The readout still never selects candidates — NLSS remains a representation
layer, not a controller.

Frozen V6.1 boundaries (per governance):
  * state, grounding, recovery, EvidenceBuilder: UNCHANGED;
  * this module is purely additive.

Leakage / no-cheat guarantees (enforced by tests):
  * the readout consumes ONLY the constructed objects' canonical descriptors
    and the adapter-provided PUBLIC task-grammar attribute domains;
  * it NEVER calls a validator, NEVER enumerates the admissible solution set,
    and NEVER proposes valid neighbors found by programmatic enumeration;
  * the LLM still produces every hypothesis and each candidate is verified
    under the normal query budget.
"""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .interfaces import TaskAdapter
    from .state import SolutionSpaceState
    from .types import GroundedObject, PosteriorView


# Half-width of the "uncertain boundary" band around p_t(h)=0.5.  Named so the
# readout never hard-codes the magic ``abs(p - 0.5) < 0.25`` literal (T3).
DEFAULT_UNCERTAIN_BAND: float = 0.25


# ---------------------------------------------------------------------------
# Frontier semantics — the benchmark-specific contract (lives in adapters)
# ---------------------------------------------------------------------------


class FrontierSemantics(Protocol):
    """Benchmark-provided structural frontier vocabulary.

    Implemented by adapters.  Everything here is outcome-blind and derived
    from the canonical object + the PUBLIC task grammar only.
    """

    def descriptor(self, obj: "GroundedObject") -> Mapping[str, Any]:
        """Canonical, outcome-blind structural descriptor of a grounded
        object (e.g. causal: (n_edges, degree profile, #v-structures, longest
        path, source/sink pattern))."""
        ...

    def attribute_domains(self) -> Mapping[str, tuple[Any, ...]]:
        """Public task-grammar value ranges per descriptor attribute
        (discretized).  The frontier readout expands cells over this grammar
        product space — never over the solution set."""
        ...

    def revision_hints(self, from_descriptor: Mapping[str, Any], to_descriptor: Mapping[str, Any]) -> tuple[str, ...]:
        """Interpretable G_rev action classes plausibly spanning this
        descriptor gap (from the public revision vocabulary; no validation)."""
        ...

    def cell_feasible(self, descriptor: Mapping[str, Any]) -> bool:
        """Whether a descriptor cell is feasible under the PUBLIC task grammar
        (readout v2: attribute consistency rules, e.g. depth>0 => n_ops>0).
        Optional; the core defaults to accepting every cell."""
        ...

    def target_instruction(self, descriptor: Mapping[str, Any]) -> str:
        """Domain-flavoured directive for one frontier target (readout v2),
        e.g. 'produce a hypothesis using at least 3 variables with an OR
        interaction'.  Optional; the core falls back to generic text."""
        ...


# ---------------------------------------------------------------------------
# Readout dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoveredRegion:
    """A connected cluster of covered descriptor cells (V_t in descriptor
    space)."""

    region_id: str
    descriptor: Mapping[str, Any]
    object_ids: tuple[str, ...]
    size: int


@dataclass(frozen=True, slots=True)
class FrontierTarget:
    """One concrete space-expansion direction: an uncovered descriptor cell
    adjacent (in descriptor space) to the covered region."""

    target_id: str
    descriptor: Mapping[str, Any]
    anchor_object_id: str | None
    attribute_changes: Mapping[str, tuple[Any, Any]]  # attr -> (from, to)
    revision_hints: tuple[str, ...]
    rationale: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SolutionFamily:
    """One section-5.3 view of the recovered solution space (V7 additive).

    The proposal distinguishes several *derived views* of the recovered
    solution-space model (not extra primitives):

      - ``supported``          : empirically supported solutions, p_t(h) >= eta;
      - ``probable_untested``  : high posterior belief but not yet observed;
      - ``family``             : a distinct solution family (a covered region
                                 cluster in descriptor space);
      - ``under_covered``      : an uncovered frontier region to expand into;
      - ``uncertain_boundary`` : high-uncertainty region (p_t(h) ~ 0.5).

    ``object_ids`` are the members of the view (empty for ``under_covered`` /
    ``uncertain_boundary``, which are descriptor regions rather than member sets).
    """

    view: str
    family_id: str
    object_ids: tuple[str, ...]
    descriptor: Mapping[str, Any]
    eta: float | None = None
    size: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReadoutPacket:
    task_id: str
    round_index: int
    covered_regions: tuple[CoveredRegion, ...]
    frontier_targets: tuple[FrontierTarget, ...]
    constructed_count: int
    covered_cell_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # V7 additive: section-5.3 solution-family views (empty when no posterior
    # was supplied to the readout).
    families: tuple[SolutionFamily, ...] = ()


# ---------------------------------------------------------------------------
# SpaceReadout — the formal component
# ---------------------------------------------------------------------------


class SpaceReadout(ABC):
    """Maps the shared SolutionSpaceState to frontier readout packets."""

    @abstractmethod
    def build(
        self,
        state: "SolutionSpaceState",
        semantics: FrontierSemantics,
    ) -> ReadoutPacket:
        """F_t from S_t: covered structural regions + uncovered frontiers."""

    def build_with_posterior(
        self,
        state: "SolutionSpaceState",
        semantics: FrontierSemantics,
        posterior: "PosteriorView | None",
        eta: float | None = None,
    ) -> ReadoutPacket:
        """V7 additive path: ``build`` plus, when a fitted ``posterior`` is
        supplied, the section-5.3 solution-family views.  The frozen ``build``
        core is unchanged; the default simply returns ``build`` (no posterior).
        The concrete readout overrides this to materialize the family views.
        """
        return self.build(state, semantics)


def descriptor_tuple(descriptor: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Stable serialization of a descriptor for hashing/sorting."""
    items = descriptor.items() if isinstance(descriptor, Mapping) else descriptor
    return tuple(sorted((str(k), v) for k, v in items))


def descriptor_distance(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> float:
    """Hamming distance over descriptor attributes (0 = same cell)."""
    keys = set(a) | set(b)
    return float(sum(1 for k in keys if a.get(k) != b.get(k)))


class CoverageFrontierReadout(SpaceReadout):
    """Generic coverage-frontier readout.

    Algorithm (deterministic given seed):
      1. covered cells = descriptors of V_t objects;
      2. boundary expansion: walk the PUBLIC grammar product space BFS-style
         from the covered cells, up to ``radius`` attribute-flips per step
         (radius 1 = immediate frontier; capped by ``max_cells``);
      3. frontier cells = reached uncovered cells at distance <= radius;
      4. rank: distance asc, then isolation (fewer covered neighbours), then
         diversity (greedy max pairwise distance);
      5. one FrontierTarget per selected cell, anchored to the nearest
         covered object, with attribute changes + revision hints.

    Never touches the validator or the admissible set.
    """

    def __init__(
        self,
        *,
        n_targets: int = 4,
        radius: int = 1,
        max_cells: int = 20000,
        seed: int = 0,
        uncertain_band: float = DEFAULT_UNCERTAIN_BAND,
    ) -> None:
        self.n_targets = n_targets
        self.radius = radius
        self.max_cells = max_cells
        self.seed = seed
        # Half-width of the "uncertain boundary" band around p=0.5 (T3 tech-
        # debt: replaces a magic literal with a named, configurable parameter).
        self.uncertain_band = float(uncertain_band)

    # -- descriptor-space helpers -------------------------------------------

    def _covered_cells(
        self, objects: Sequence["GroundedObject"], semantics: FrontierSemantics
    ) -> dict[tuple, list[str]]:
        cells: dict[tuple, list[str]] = {}
        for obj in objects:
            cells.setdefault(descriptor_tuple(semantics.descriptor(obj)), []).append(obj.object_id)
        return cells

    def _frontier_cells(
        self,
        domains: Mapping[str, tuple[Any, ...]],
        covered: dict[tuple, list[str]],
        semantics: FrontierSemantics | None = None,
    ) -> list[tuple[tuple, int]]:
        """Uncovered cells at distance <= radius from a covered cell, with
        their distance.  BFS boundary expansion over the grammar product;
        readout-v2 feasibility filter keeps only cells consistent with the
        public task grammar."""
        attrs = sorted(domains)
        has_feasibility = semantics is not None and getattr(semantics, "cell_feasible", None) is not None
        frontier: dict[tuple, int] = {}
        queue: list[tuple[tuple, int]] = [(cell, 0) for cell in covered]
        seen: set[tuple] = set(covered)
        head = 0
        while head < len(queue) and len(frontier) < self.max_cells:
            cell, dist = queue[head]
            head += 1
            if dist >= self.radius:
                continue
            cell_map = dict(cell)
            for a in attrs:
                for value in domains[a]:
                    if value == cell_map.get(a):
                        continue
                    nxt = dict(cell_map)
                    nxt[a] = value
                    nxt_tuple = descriptor_tuple(nxt)
                    if nxt_tuple in seen:
                        continue
                    seen.add(nxt_tuple)
                    if has_feasibility and not semantics.cell_feasible(nxt):
                        continue
                    if nxt_tuple in covered:
                        queue.append((nxt_tuple, dist + 1))
                    else:
                        frontier[nxt_tuple] = dist + 1
                        queue.append((nxt_tuple, dist + 1))
        return [(cell, d) for cell, d in frontier.items()]

    # -- build ---------------------------------------------------------------

    def build(self, state: "SolutionSpaceState", semantics: FrontierSemantics) -> ReadoutPacket:
        objects = tuple(state.objects.values())
        covered = self._covered_cells(objects, semantics)
        domains = semantics.attribute_domains()
        frontier = self._frontier_cells(domains, covered, semantics)

        def isolation(cell: tuple) -> int:
            return sum(
                1
                for c in covered
                if descriptor_distance(dict(cell), dict(c)) <= 2 * self.radius
            )

        scored = sorted(frontier, key=lambda t: (t[1], isolation(t[0])))

        # diversity selection (greedy, deterministic)
        selected: list[tuple] = []
        for cell, _dist in scored:
            if len(selected) >= self.n_targets:
                break
            if all(descriptor_distance(dict(cell), dict(s)) > 0 for s in selected):
                selected.append(cell)

        # covered regions: connected components of covered cells (distance <= 1)
        covered_cell_list = [dict(c) for c in covered]
        regions: list[CoveredRegion] = []
        remaining = list(range(len(covered_cell_list)))
        region_idx = 0
        while remaining:
            start = remaining.pop(0)
            comp = [covered_cell_list[start]]
            queue = [start]
            while queue:
                i = queue.pop(0)
                for j in list(remaining):
                    if descriptor_distance(covered_cell_list[i], covered_cell_list[j]) <= 1.0:
                        remaining.remove(j)
                        queue.append(j)
                        comp.append(covered_cell_list[j])
            rep = comp[0]
            member_ids: list[str] = []
            for cell in comp:
                member_ids.extend(covered[descriptor_tuple(cell)])
            regions.append(
                CoveredRegion(
                    region_id=f"region_{region_idx}",
                    descriptor=rep,
                    object_ids=tuple(sorted(set(member_ids))),
                    size=len(set(member_ids)),
                )
            )
            region_idx += 1

        targets: list[FrontierTarget] = []
        for idx, cell in enumerate(selected):
            nearest_tuple = min(covered, key=lambda c: descriptor_distance(dict(cell), dict(c)))
            anchor_id = covered[nearest_tuple][0]
            nearest_map = dict(nearest_tuple)
            changes = {
                a: (nearest_map.get(a), cell_map.get(a))
                for a, cell_map in [("_", dict(cell))]
                for a in cell_map
                if nearest_map.get(a) != cell_map.get(a)
            }
            hints = semantics.revision_hints(nearest_map, dict(cell))
            rationale = (
                f"descriptor cell {descriptor_tuple(dict(cell))} has 0 constructed "
                f"objects; nearest covered cell differs in {len(changes)} attribute(s)"
            )
            targets.append(
                FrontierTarget(
                    target_id=f"frontier_{idx}",
                    descriptor=dict(cell),
                    anchor_object_id=anchor_id,
                    attribute_changes=changes,
                    revision_hints=hints,
                    rationale=rationale,
                )
            )

        return ReadoutPacket(
            task_id=state.task_id,
            round_index=state.round_index,
            covered_regions=tuple(regions),
            frontier_targets=tuple(targets),
            constructed_count=len(objects),
            covered_cell_count=len(covered),
            metadata={"readout": "coverage_frontier", "radius": self.radius, "seed": self.seed},
        )

    def build_with_posterior(
        self,
        state: "SolutionSpaceState",
        semantics: FrontierSemantics,
        posterior: "PosteriorView | None",
        eta: float | None = None,
    ) -> ReadoutPacket:
        """V7 additive path: ``build`` plus section-5.3 solution-family views.

        When a fitted ``posterior`` is available, the readout (which normally
        only sees structural coverage) is augmented with the posterior-driven
        family views of the recovered solution space: supported / probable-
        untested / distinct families / under-covered / uncertain boundaries.
        ``eta`` (the probability cutoff, T3) defaults to
        ``state.metadata["eta"]`` (the facade's eta) or :data:`DEFAULT_ETA`
        (0.5) when absent — it is NOT the backend's gamma value cut.
        Pure ``build`` (frozen core) is left untouched.
        """
        packet = self.build(state, semantics)
        if posterior is None:
            return packet
        if eta is None:
            eta = float(state.metadata.get("eta", 0.5) or 0.5)
        families = self._classify_families(state, semantics, posterior, packet, eta)
        return replace(
            packet,
            families=tuple(families),
            metadata={**packet.metadata, "n_families": len(families)},
        )

    def _classify_families(
        self,
        state: "SolutionSpaceState",
        semantics: FrontierSemantics,
        posterior: "PosteriorView",
        packet: ReadoutPacket,
        eta: float,
    ) -> list[SolutionFamily]:
        """Section-5.3 four-ish perspectives from the posterior marginals.

        The proposal says these are *views* of the recovered solution-space
        model, so they are derived here from ``posterior.marginal`` (the true
        probability belief p_t(h)) plus the already-computed coverage structure.
        """
        objects = state.objects
        observed = set(state.observations)
        marginals: dict[str, Any] = {}
        for oid in objects:
            try:
                marginals[oid] = posterior.marginal(oid)
            except Exception:
                marginals[oid] = None

        def _sol_prob(oid: str) -> float | None:
            m = marginals[oid]
            if m is None or m.solution_probability is None:
                return None
            return float(m.solution_probability)

        out: list[SolutionFamily] = []

        # 1) empirically supported solutions
        supported = [oid for oid in objects if _sol_prob(oid) is not None and _sol_prob(oid) >= eta]
        if supported:
            out.append(
                SolutionFamily(
                    view="supported", family_id="supported",
                    object_ids=tuple(supported), descriptor={},
                    eta=eta, size=len(supported),
                    metadata={"count": len(supported)},
                )
            )

        # 2) probable untested alternatives
        untested = [
            oid for oid in objects
            if oid not in observed and _sol_prob(oid) is not None and _sol_prob(oid) >= eta
        ]
        if untested:
            out.append(
                SolutionFamily(
                    view="probable_untested", family_id="probable_untested",
                    object_ids=tuple(untested), descriptor={},
                    eta=eta, size=len(untested),
                    metadata={"count": len(untested)},
                )
            )

        # 3) distinct solution families == the covered-region clusters
        for region in packet.covered_regions:
            out.append(
                SolutionFamily(
                    view="family", family_id=region.region_id,
                    object_ids=region.object_ids, descriptor=dict(region.descriptor),
                    eta=eta, size=region.size,
                )
            )

        # 4) under-covered parts == the uncovered frontier targets
        if packet.frontier_targets:
            out.append(
                SolutionFamily(
                    view="under_covered", family_id="under_covered",
                    object_ids=tuple(t.target_id for t in packet.frontier_targets),
                    descriptor={}, eta=eta, size=len(packet.frontier_targets),
                    metadata={"count": len(packet.frontier_targets)},
                )
            )

        # 5) uncertain boundaries == p_t(h) near 0.5 (named ``uncertain_band``
        #    half-width, T3 — no magic literal).
        band = self.uncertain_band
        uncertain = [
            oid for oid in objects
            if _sol_prob(oid) is not None and abs(_sol_prob(oid) - 0.5) < band
        ]
        if uncertain:
            out.append(
                SolutionFamily(
                    view="uncertain_boundary", family_id="uncertain_boundary",
                    object_ids=tuple(uncertain), descriptor={},
                    eta=eta, size=len(uncertain),
                    metadata={"count": len(uncertain)},
                )
            )

        return out


# ---------------------------------------------------------------------------
# Rendering (token-budgeted text)
# ---------------------------------------------------------------------------


def render_readout(
    packet: ReadoutPacket,
    state: "SolutionSpaceState",
    adapter: "TaskAdapter",
    token_budget: int,
) -> str:
    """Render a ReadoutPacket into a prompt block within ``token_budget``.

    Readout v2: directive phrasing — each frontier target is an explicit
    instruction the model can follow, plus the canonical non-repeat clause.
    """
    lines: list[str] = []
    lines.append("## Constructed structural coverage (descriptor map)")
    lines.append(
        f"- {packet.constructed_count} constructed objects, "
        f"{packet.covered_cell_count} distinct structural cells, "
        f"{len(packet.covered_regions)} covered regions"
    )
    lines.append("## Covered structural regions")
    for region in packet.covered_regions[:6]:
        reps = []
        for oid in region.object_ids[:4]:
            obj = state.objects.get(oid)
            reps.append(adapter.render_object(obj) if obj else oid)
        desc = ", ".join(f"{k}={v}" for k, v in sorted(region.descriptor.items()))
        lines.append(f"- region ({region.size}): {desc}; members: {', '.join(reps)}")
    lines.append("## Uncovered frontiers — next space-expansion targets")
    lines.append(
        "Produce a hypothesis that satisfies the observations AND whose "
        "canonical form differs from every anchored object below."
    )
    for i, t in enumerate(packet.frontier_targets, start=1):
        anchor = t.anchor_object_id or "none"
        if anchor in state.objects:
            anchor = adapter.render_object(state.objects[anchor])
        changes = "; ".join(f"{a}: {f}->{t2}" for a, (f, t2) in sorted(t.attribute_changes.items()))
        hints = ", ".join(t.revision_hints)
        instruction = getattr(adapter, "target_instruction", None)
        if instruction is not None:
            try:
                directive = instruction(t.descriptor)
            except Exception:
                directive = ""
        else:
            directive = ""
        lines.append(
            f"- Target {i}: {directive} "
            f"(descriptor {{ {', '.join(f'{k}={v}' for k, v in sorted(t.descriptor.items()))} }}; "
            f"differs from anchor {anchor}: {changes}; revision classes: {hints})"
        )
    text = "\n".join(lines)
    return text[: token_budget * 4]
