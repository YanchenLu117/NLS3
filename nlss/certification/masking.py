"""V8 P0-d — certification primitives (Detail §4.1/§4.2).

Evaluator-side finite-model machinery for HypoSpace Level-2:

- :class:`EvaluatorUniverse` — the complete admissible semantic universe
  W^eval (evaluator-only).  Generators carry public deterministic predicates;
  sigma(U) is the bit mask of worlds satisfying U.
- :func:`enumerate_terms` — canonical monotone normal-form terms over G_t
  (irredundant antichain DNF, including bottom and top), with hard size/time
  limits; exceeding either raises :class:`CertificationIncomplete` — never an
  approximate result (Detail §4.1).
- :func:`build_image_frame` — merge equal-mask terms into the quotient image
  frame; image IDs are assigned in deterministic lexicographic bit-mask order.

The runtime uses ordinary sets/bitmask operations; no category machinery.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

CERTIFICATION_INCOMPLETE = "CERTIFICATION_INCOMPLETE"

# A canonical monotone DNF term: irredundant antichain of nonempty generator
# index sets (bottom = empty antichain; top = {∅}).
TermAntichain = frozenset[frozenset[int]]


def antichain(subsets: Iterable[frozenset[int]]) -> TermAntichain:
    """Reduce a family of index sets to its antichain (drop empty and supersets)."""
    family = {s for s in subsets if s}
    minimal: list[frozenset[int]] = []
    for s in sorted(family, key=len):
        if not any(m < s or m == s for m in minimal):
            minimal.append(s)
    return frozenset(minimal)


class CertificationIncomplete(RuntimeError):
    """Term enumeration or image closure exceeded P0 size/time limits.

    Detail §4.1: exceeding either limit yields CERTIFICATION_INCOMPLETE, not an
    approximate Level-2 result.
    """


@dataclass(frozen=True)
class EvaluatorUniverse:
    """W^eval with deterministic world ordering (sorted ids)."""

    world_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.world_ids)) != len(self.world_ids):
            raise ValueError("world ids must be unique")

    @property
    def size(self) -> int:
        return len(self.world_ids)

    def full_mask(self) -> int:
        return (1 << len(self.world_ids)) - 1

    def sigma(self, predicate: Callable[[str], bool]) -> int:
        """Bit mask of worlds satisfying the predicate (bit i = world_ids[i])."""
        mask = 0
        for i, wid in enumerate(self.world_ids):
            if predicate(wid):
                mask |= 1 << i
        return mask


# --------------------------------------------------------------------------
# Canonical monotone normal-form terms (irredundant antichain DNF)
# --------------------------------------------------------------------------


def _antichain(subsets: Iterable[frozenset[int]]) -> frozenset[frozenset[int]]:
    """Reduce a family of nonempty index sets to its antichain (drop supersets)."""
    family = {s for s in subsets if s}
    minimal: list[frozenset[int]] = []
    for s in sorted(family, key=len):
        if not any(m < s or m == s for m in minimal):
            minimal.append(s)
    return frozenset(minimal)


def enumerate_terms(n_generators: int, *, max_terms: int, max_seconds: float = 10.0) -> tuple[frozenset[frozenset[frozenset[int]]], bool]:
    """All canonical monotone DNF terms over n generators, including bottom
    (empty antichain) and top (the antichain containing the empty conjunction).

    Terms are irredundant antichains of nonempty index subsets, plus the two
    constants: bottom = empty antichain; top = antichain {∅} (an empty
    conjunction is true).  For k generators this yields the free bounded
    distributive lattice M(k) elements (M(2)=6, M(3)=20, M(4)=168, M(5)=7581).
    Raises :class:`CertificationIncomplete` when the P0 ``max_terms`` limit or
    ``max_seconds`` is exceeded.
    """
    deadline = time.monotonic() + max_seconds
    if n_generators < 1:
        raise ValueError("need at least one generator")
    top_term = frozenset({frozenset()})

    terms: set[frozenset[frozenset[int]]] = set()
    all_subsets: list[frozenset[int]] = []
    for bits in range(1, 1 << n_generators):
        s = frozenset(i for i in range(n_generators) if bits >> i & 1)
        all_subsets.append(s)

    def dfs(next_idx: int, chosen: list[frozenset[int]]) -> Iterator[frozenset[frozenset[int]]]:
        if time.monotonic() > deadline:
            raise CertificationIncomplete(f"term enumeration exceeded {max_seconds:.1f}s")
        yield _antichain(chosen)
        for j in range(next_idx, len(all_subsets)):
            s = all_subsets[j]
            if any(c <= s for c in chosen):
                continue
            yield from dfs(j + 1, chosen + [s])

    count = 0
    for term in dfs(0, []):
        terms.add(term)
        count += 1
        if count > max_terms:
            raise CertificationIncomplete(f"term enumeration exceeded max_terms={max_terms}")
    terms.add(top_term)
    return frozenset(terms), True


def term_mask(term: frozenset[frozenset[int]], generator_masks: Mapping[int, int], width: int) -> int:
    """Semantic mask of a term: OR over members of (AND over generator masks).

    bottom (empty antichain) -> 0; top ({∅}) -> full mask.
    """
    full = (1 << width) - 1
    if not term:
        return 0
    if frozenset() in term:
        return full
    mask = 0
    for member in term:
        m = full
        for idx in member:
            m &= generator_masks[idx]
        mask |= m
    return mask


@dataclass(frozen=True)
class ImageFrame:
    """Quotient image frame: distinct masks with deterministic lexicographic ids."""

    width: int
    masks: tuple[int, ...]  # ascending

    @property
    def size(self) -> int:
        return len(self.masks)

    def image_id(self, mask: int) -> str:
        return format(mask, f"0{self.width}b")

    @property
    def bottom(self) -> int:
        return 0

    @property
    def top(self) -> int:
        return (1 << self.width) - 1

    def meet(self, a: int, b: int) -> int:
        return a & b

    def join(self, a: int, b: int) -> int:
        return a | b

    def is_refinement(self, a: int, b: int) -> bool:
        """a ⊑ b iff mask(a) ⊆ mask(b)."""
        return a & b == a

    def covers(self, parent: int, parts: Sequence[int]) -> bool:
        """{parts} covers parent iff bitwise OR of parts equals parent mask."""
        acc = 0
        for p in parts:
            acc |= p
        return acc == parent

    def index(self, mask: int) -> int:
        try:
            return self.masks.index(mask)
        except ValueError:
            raise KeyError(f"mask {mask:#b} not in frame")


def build_image_frame(
    universe: EvaluatorUniverse,
    generator_masks: Mapping[int, int],
    *,
    max_seconds: float = 10.0,
) -> ImageFrame:
    """Close {generator masks, bottom, top} under meet/join to the fixpoint and
    merge equal masks (distinct masks = image ids in ascending lexicographic
    bit-mask order).  Worklist incremental closure: each new mask is combined
    only with the current element set — total O(n²) instead of full rescans."""
    deadline = time.monotonic() + max_seconds
    full = universe.full_mask()
    current: set[int] = {0, full, *generator_masks.values()}
    worklist: list[int] = sorted(current)
    while worklist:
        if time.monotonic() > deadline:
            raise CertificationIncomplete(f"image closure exceeded {max_seconds:.1f}s")
        new_masks: list[int] = []
        snapshot = list(current)  # current grows during the pass — iterate frozen
        for m in worklist:
            for e in snapshot:
                for value in (m & e, m | e):
                    if value not in current:
                        current.add(value)
                        new_masks.append(value)
        worklist = new_masks
    return ImageFrame(width=universe.size, masks=tuple(sorted(current)))


def family_separation(
    universe: EvaluatorUniverse,
    generator_predicates: Mapping[str, Callable[[str], bool]],
    official_families: Sequence[Callable[[str], bool]],
) -> tuple[float, int, int]:
    """FamilySep_t (Detail §4.2): partition W^eval by generator truth signatures;
    a family is separated when some nonempty cell is contained in it.

    Returns (family_sep, separated_count, official_count).
    """
    if not official_families:
        return 1.0, 0, 0
    ids = sorted(generator_predicates)
    cells: dict[tuple[bool, ...], set[str]] = {}
    for wid in universe.world_ids:
        sig = tuple(bool(generator_predicates[g](wid)) for g in ids)
        cells.setdefault(sig, set()).add(wid)
    separated = 0
    for family in official_families:
        for cell in cells.values():
            if cell and all(family(wid) for wid in cell):
                separated += 1
                break
    return separated / len(official_families), separated, len(official_families)
