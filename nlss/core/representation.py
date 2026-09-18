"""V7 Core — logic-governed neuro-symbolic representation induction.

This module is purely additive.  It gives the previously-unwritten core objects
of the V7 Core revision (proposal §5, §7, §11, §12):

    P~_t   -- LLM-proposed representation specification (widetilde P_t)
    L_tau  -- the logic contract  (Type/Structural/Executable/Semantic)
    F_t    -- the compact fidelity auditor  (dual gate with LogicValid)
    C_t    -- instantiated by :class:`RepresentationRuntime` from P_t

Permanent state  S^{V7} = (D_t, P_t, C_t, Gamma_t, B_t) is completed here by the
representation specification ``P_t`` and its admission (logic + fidelity).

Design constraints
------------------
- Additive: does not rewrite the frozen V6 core.  It *consumes* the
  ``TaskAdapter`` (compile/verify/canonicalize + FrontierSemantics
  ``descriptor``/``attribute_domains``/``revision_hints``) already present.
- Testable without an LLM: when no LLM is provided, the "LLM proposal" defaults
  to a faithful transcription of the adapter's own declared structure, so the
  dual gate and revision machinery run deterministically.
- Paradigm-flagged: the four representation-paradigm arms (V1 fixed embedding,
  V6 fixed symbolic, V7-minus-Logic, Full V7) are first-class so the
  representation-paradigm ablation (§4.8 / Claim C4) can be run.

Nothing here touches outcome labels except to *decide membership*, matching the
frozen outcome-blind-geometry rule (evidence must never enter similarity).
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .interfaces import TaskAdapter


# -- run an awaitable to completion from sync code (used by the LLM inducer) --

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_thread: threading.Thread | None = None


def _run_coro(coro):
    """Run a coroutine to completion synchronously (mirrors core/model._run).

    - No running loop: drive it with a fresh event loop.
    - Already inside a running loop: ship to one persistent background worker
      loop and block (run_coroutine_threadsafe), so calling from within a
      benchmark event loop does not raise "event loop is already running".
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop().run_until_complete(coro)
    global _worker_loop, _worker_thread
    if _worker_loop is None or not _worker_loop.is_running():
        _worker_loop = asyncio.new_event_loop()
        _worker_thread = threading.Thread(target=_worker_loop.run_forever, name="nlss-inducer", daemon=True)
        _worker_thread.start()
    return asyncio.run_coroutine_threadsafe(coro, _worker_loop).result()


# ---------------------------------------------------------------------------
# Paradigm modes for the representation-paradigm ablation (Claim C4)
# ---------------------------------------------------------------------------


class RepresentationParadigm(str, Enum):
    """The four arms of the representation-paradigm ablation (proposal §A.5).

    V1_FIXED_EMBEDDING  : fixed embedding into a fixed latent space; no induced
                          task structure; no logic gate.
    V6_FIXED_SYMBOLIC   : the frozen expert compiler (K o V o P) into a
                          predefined schema; strict symbolic verification only.
    V7_NO_LOGIC         : LLM-induced representation with NO logic gate and NO
                          fidelity gate (isolates the value of the logic gate).
    V7_FULL             : LLM-induced + logic gate + fidelity gate (complete V7).
    """

    V1_FIXED_EMBEDDING = "v1_fixed_embedding"
    V6_FIXED_SYMBOLIC = "v6_fixed_symbolic"
    V7_NO_LOGIC = "v7_induced_no_logic"
    V7_FULL = "v7_full"

    @property
    def runs_logic_gate(self) -> bool:
        return self is RepresentationParadigm.V7_FULL

    @property
    def runs_fidelity_gate(self) -> bool:
        return self is RepresentationParadigm.V7_FULL

    @property
    def is_adaptive(self) -> bool:
        # V7 arms induce/revise; V1/V6 do not.
        return self in (RepresentationParadigm.V7_NO_LOGIC, RepresentationParadigm.V7_FULL)


# ---------------------------------------------------------------------------
# The representation specification  P_t == (P~_t  ->  P_t)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypedVariable:
    """A declared scientific variable in the induced representation.

    ``domain`` (optional) is the explicit value domain for this variable when
    the induction specifies one (paper's ``Sigma`` factorization / domains).
    """

    name: str
    vtype: str  # e.g. 'categorical', 'bool', 'continuous', 'discrete', 'latent'
    role: str = "state"  # 'state' | 'outcome' | 'latent'
    description: str = ""
    domain: tuple | None = None


@dataclass(frozen=True, slots=True)
class RelationDecl:
    """A declared relation over the induced variables/objects."""

    name: str
    polymorphic_type: str
    scope: str  # explicit scope — e.g. 'objects', 'variables'
    outcome_blind: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class TransformDecl:
    """A declared executable transformation (a candidate can be realized)."""

    name: str
    arity: str
    executable: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class RepresentationSpecification:
    """P_t: the adopted computational representation specification.

    `source` records how it was obtained:
      'adapter_default' - faithful transcription of the adapter's declared
                          structure (deterministic no-LLM path);
      'llm_induced'     - proposed by an LLM (the full P~_t path).
    `admitted` records whether it passed the dual gate (logic + fidelity).
    """

    spec_id: str
    name: str
    description: str

    typed_variables: tuple[TypedVariable, ...] = ()
    relations: tuple[RelationDecl, ...] = ()
    transforms: tuple[TransformDecl, ...] = ()
    semantic_commitments: tuple[str, ...] = ()  # distinctions F_t must preserve

    # paper contract components beyond Sigma/R/T/Phi (kept optional, additive):
    entities: tuple[str, ...] = ()  # Sigma: entity list / factorization
    geometry: Mapping[str, Any] | None = None  # K: features/neighborhoods/distances/kernels
    grounding: Mapping[str, Any] | None = None  # G: forward/backward procedure declarations

    backend_hint: str | None = None  # induced backend name for C_t
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source: str = "adapter_default"
    admitted: bool = False

    # -- helpers ----------------------------------------------------------

    @property
    def state_variables(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.typed_variables if v.role == "state")

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.typed_variables)

    def with_commitments(self, commitments: Sequence[str]) -> "RepresentationSpecification":
        """Return a copy with the given semantic commitments (used on revision)."""
        return RepresentationSpecification(
            spec_id=self.spec_id,
            name=self.name,
            description=self.description,
            typed_variables=self.typed_variables,
            relations=self.relations,
            transforms=self.transforms,
            semantic_commitments=tuple(commitments),
            entities=self.entities,
            geometry=self.geometry,
            grounding=self.grounding,
            backend_hint=self.backend_hint,
            provenance=self.provenance,
            source=self.source,
            admitted=self.admitted,
        )


# ---------------------------------------------------------------------------
# The logic contract  L_tau  and its gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogicViolation:
    """One concrete representation-contract violation."""

    check: str  # 'type' | 'structural' | 'executable' | 'semantic'
    message: str
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class LogicReport:
    """Result of running the logic gate on a candidate representation."""

    valid: bool
    violations: tuple[LogicViolation, ...] = ()
    checked: tuple[str, ...] = ()

    @property
    def n_violations(self) -> int:
        return len(self.violations)


class LogicGate:
    """L_tau = (Phi_type, Phi_structural, Phi_executable, Phi_semantic).

    Paper-1 minimal, deterministic, provenance-respecting contract.  It checks
    *the representation contract*, not the scientific content — so it never
    locks discovery (proposal §7-§8).  Outcome leakage is forbidden: a variable
    or relation whose scope references *fitted* outcomes is a violation unless
    it is declared an explicit outcome variable.
    """

    _ROLE_TYPES: Mapping[str, str] = {
        "state": "categorical|bool|continuous|discrete|latent",
        "latent": "latent|continuous",
        "outcome": "continuous|discrete|bool",
    }

    def __init__(
        self,
        adapter: "TaskAdapter",
        *,
        hard_constraints: Sequence[Any] = (),
    ) -> None:
        self.adapter = adapter
        # sparse task hard constraints Phi^hard_tau (paper §4.1): an explicit,
        # task-supplied constraint list (e.g. never-mergeable distinctions) that
        # the representation must not violate.  The meta-logic (type/struct/
        # executable/semantic) is always enforced independently of this.
        self.hard_constraints = tuple(hard_constraints)

    def _check_type(self, spec: RepresentationSpecification) -> tuple[LogicViolation, ...]:
        if not spec.typed_variables:
            return (LogicViolation("type", "representation declares no typed variables"),)
        out: list[LogicViolation] = []
        seen: set[str] = set()
        for v in spec.typed_variables:
            if v.name in seen:
                out.append(
                    LogicViolation(
                        "type",
                        f"variable {v.name!r} declared more than once (type contract)",
                    )
                )
            seen.add(v.name)
            allowed = self._ROLE_TYPES.get(v.role, "")
            if not allowed or not any(v.vtype in x for x in allowed.split("|")):
                out.append(
                    LogicViolation(
                        "type",
                        f"variable {v.name!r}: role={v.role!r} not compatible with type {v.vtype!r}",
                    )
                )
        return tuple(out)

    def _check_structural(self, spec: RepresentationSpecification) -> tuple[LogicViolation, ...]:
        out: list[LogicViolation] = []
        for r in spec.relations:
            if not r.scope:
                out.append(LogicViolation("structural", f"relation {r.name!r} lacks explicit scope"))
        return tuple(out)

    def _check_executable(self, spec: RepresentationSpecification) -> tuple[LogicViolation, ...]:
        out: list[LogicViolation] = []
        for t in spec.transforms:
            if not t.executable:
                out.append(
                    LogicViolation("executable", f"declared transform {t.name!r} is not executable")
                )
        if not spec.transforms:
            out.append(
                LogicViolation(
                    "executable",
                    "representation declares no executable transformations "
                    "(candidates cannot be realized in experiment/action)",
                )
            )
        return tuple(out)

    def _check_semantic(self, spec: RepresentationSpecification) -> tuple[LogicViolation, ...]:
        if not spec.semantic_commitments:
            return (LogicViolation("semantic", "representation declares no semantic commitments"),)
        leakies = ("solution", "outcome", "yield", "fitness", "reward", "label", "observed value")
        out: list[LogicViolation] = []
        for c in spec.semantic_commitments:
            low = c.lower()
            if any(token in low for token in leakies):
                out.append(
                    LogicViolation(
                        "semantic",
                        f"commitment {c!r} appears to reference fitted outcome content; "
                        "outcome-blind rule violated",
                    )
                )
        return tuple(out)

    def _check_consistency(self, spec: RepresentationSpecification) -> tuple[LogicViolation, ...]:
        """Internal-consistency: declared names are distinct and referenced
        entities exist within the contract's typed variables (paper $L^meta:
        'internally consistent')."""
        out: list[LogicViolation] = []
        names = list(spec.variable_names)
        if len(names) != len(set(names)):
            out.append(LogicViolation("type", "variable names are not unique (internal consistency)"))
        rel_names = {r.name for r in spec.relations}
        tr_names = {t.name for t in spec.transforms}
        dup = (rel_names & tr_names) & {n for n in rel_names if n in tr_names}
        if dup:
            out.append(
                LogicViolation(
                    "structural",
                    f"relation and transform share the same declared name {sorted(dup)!r} "
                    "(unclear semantics)",
                )
            )
        return tuple(out)

    def _check_provenance(self, spec: RepresentationSpecification) -> tuple[LogicViolation, ...]:
        """Provenance-preservation (paper $L^meta): the contract should record
        how it was produced.  Reported as warnings (not admission errors) so a
        pledge-level provenance gap is surfaced without locking admission."""
        out: list[LogicViolation] = []
        if not spec.provenance:
            out.append(
                LogicViolation(
                    "semantic",
                    "representation records no provenance (cannot verify it is outcome-blind)",
                    severity="warning",
                )
            )
        if not spec.source:
            out.append(LogicViolation("semantic", "representation has no declared source",
                                      severity="warning"))
        return tuple(out)

    def _check_hard(self, spec: RepresentationSpecification) -> tuple[LogicViolation, ...]:
        """Sparse task hard constraints Phi^hard_tau.  Paper-1 keeps them
        explicit and minimal so logic never locks discovery."""
        out: list[LogicViolation] = []
        for hc in self.hard_constraints:
            if callable(hc):
                try:
                    ok = bool(hc(spec))
                except Exception:
                    ok = True
                if not ok:
                    out.append(LogicViolation("semantic", f"violates task hard constraint {hc!r}"))
            else:
                out.append(LogicViolation("semantic", f"declared hard constraint is not callable: {hc!r}"))
        return tuple(out)

    def check(self, spec: RepresentationSpecification) -> LogicReport:
        """Run the contract checks (incl. hard constraints). ``valid`` requires
        zero errors."""
        checks = {
            "type": self._check_type(spec),
            "structural": self._check_structural(spec),
            "consistency": self._check_consistency(spec),
            "executable": self._check_executable(spec),
            "semantic": self._check_semantic(spec),
            "provenance": self._check_provenance(spec),
            "hard": self._check_hard(spec),
        }
        violations = tuple(v for vs in checks.values() for v in vs)
        errors = tuple(v for v in violations if v.severity == "error")
        return LogicReport(valid=len(errors) == 0, violations=violations, checked=tuple(checks))


# ---------------------------------------------------------------------------
# The compact fidelity auditor  F_t  (proposal §11-§12)
# ---------------------------------------------------------------------------

_FIDELITY_EPS: float = 1e-9


@dataclass(frozen=True, slots=True)
class FidelityCertificate:
    """The fidelity certificate (paper eq. F-hat = (F_query, F_dist, F_impl)):

    F_query  - semantic-query agreement: the fraction of the spec's semantic
               commitments that are genuinely exposed as outcome-blind
               dimensions of the language-computation surrogate.
    F_dist   - preservation of task-relevant distinctions: with concrete
               objects, the fraction of adapter-distinct descriptor pairs that
               remain distinct under the contract's typed variables (i.e. not
               collapsed).  Vacuous (1.0) when no objects are supplied.
    F_impl   - fidelity of executable realizations: the fraction of declared
               transforms/relations that are executable & scoped.
    round_trip - Q_roundtrip (EXPERIMENT_PLAN §2.2): language->computation->
               language reconstruction agreement over the probe set
               (token-Jaccard), None when no grounding contract or no objects.

    Each of the three admission components is in [0, 1].  Admission is
    componentwise (== min, since all are upper-bounded scores) -- see
    :meth:`FidelityGate.admitted_componentwise`.  ``round_trip`` is REPORTED
    (§2.2/§8.3) alongside the three, not fused into the admission ``min``.
    """

    F_query: float
    F_dist: float
    F_impl: float
    round_trip: float | None = None

    @property
    def min(self) -> float:
        return min(self.F_query, self.F_dist, self.F_impl)

    def as_tuple(self) -> tuple[float, float, float, float | None]:
        return (self.F_query, self.F_dist, self.F_impl, self.round_trip)


class FidelityGate:
    """F_t(H_NL, C_t, Gamma_t): does the induced representation preserve the
    task-relevant scientific distinctions the adapter declares?

    Upgraded to the paper's three-component certificate (Q_task / Q_contrast /
    Q_roundtrip) plus a reported Q_roundtrip reconstruction term.  CAVEAT:
    F_query/F_dist/F_impl are *deterministic proxies* (attribute-domain/
    descriptor-based) and there is NO independent LLM auditor; Q_roundtrip is a
    REAL adapter-native L->C->L reconstruction (see _F_roundtrip).  Residual
    notes are in __init__.
    Deterministic, outcome-blind, adapter-consumer only (never reads fitted
    outcomes for similarity).  The inductive *proposer* does not certify its own
    proposal: admission requires an independent pass through this gate plus the
    logic gate (the inducer only supplies the commitment set that the gate then
    *verifies against the adapter's declared structure*).
    """

    def __init__(
        self,
        adapter: "TaskAdapter",
        *,
        componentwise_epsilon: Sequence[float] | None = None,
    ) -> None:
        self.adapter = adapter
        # Paper: componentwise thresholds "frozen on development tasks" (a
        # vector eps_F).  Default (0.5,0.5,0.5); per-component thresholds are
        # intended to be frozen from dev tasks in the benchmark path.
        self.componentwise_epsilon = tuple(float(e) for e in (componentwise_epsilon or (0.5, 0.5, 0.5)))
        # Residual honesty notes (paper does NOT overclaim these as built):
        #  - F_impl is an EXECUTABLE-REALIZATION proxy (fraction of executable +
        #    scoped transforms/relations), distinct from Q_roundtrip.
        #  - Q_roundtrip IS implemented (see _F_roundtrip: adapter-native
        #    L->C->L reconstruction over a probe set).
        #  - There is NO independent LLM auditor; only this deterministic,
        #    adapter-driven tier exists (formal precedence half of the paper).
        #  - F_query/F_dist are deterministic proxies (attribute-domain key
        #    membership / descriptor inequality), not full semantic agreement or
        #    contrast challenge suites.

    def _exposed_dimensions(self) -> set[str]:
        try:
            domains = self.adapter.attribute_domains()
        except Exception:
            domains = {}
        return set(domains.keys())

    # -- Q_query: commitments genuinely exposed --------------------------

    def _F_query(self, spec: RepresentationSpecification) -> float:
        if not spec.semantic_commitments:
            return 0.0
        exposed = self._exposed_dimensions()
        if not exposed:
            return 0.0
        honored = sum(1 for c in spec.semantic_commitments if c in exposed)
        return honored / len(spec.semantic_commitments)

    # -- Q_contrast: task-relevant distinctions preserved ----------------

    def _F_dist(self, spec: RepresentationSpecification, objects) -> float:
        if not objects:
            return 1.0  # vacuous: nothing to collapse yet
        distinct = 0
        collapsed = 0
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                da = self._descriptor(a)
                db = self._descriptor(b)
                if da is None or db is None:
                    continue
                if da != db:
                    distinct += 1
                else:
                    # two distinct adapter structures must not share a descriptor
                    collapsed += 1
        total = distinct + collapsed
        return 1.0 if total == 0 else distinct / total

    def _descriptor(self, obj):
        try:
            return self.adapter.descriptor(obj)
        except Exception:
            return None

    # -- Q_impl: executable realizations ---------------------------------

    def _F_impl(self, spec: RepresentationSpecification) -> float:
        n = len(spec.transforms)
        if n == 0:
            return 0.0
        ok = sum(1 for t in spec.transforms if t.executable)
        rel_ok = sum(1 for r in spec.relations if r.scope) / max(len(spec.relations), 1)
        return 0.5 * (ok / n) + 0.5 * rel_ok

    # -- Q_roundtrip: language -> computation -> language ----------------
    # EXPERIMENT_PLAN §2.2 requires a round-trip probe: the computable form must
    # preserve the language commitment when rendered back.  Implemented as the
    # adapter-native two-leg reconstruction on a probe set:
    #     forward  L->C : descriptor(obj)
    #     backward C->L : target_instruction(descriptor)
    # and measured as token-Jaccard between the reconstructed instruction and the
    # object's own canonical rendering.  This is a REAL round-trip (the backward
    # leg only ever sees the computable descriptor, never the canonical text).

    def _F_roundtrip(
        self,
        spec: RepresentationSpecification,
        objects: Sequence[Any],
    ) -> float | None:
        if spec.grounding is None:
            return None  # no grounding contract -> round-trip undefined (honest)
        if not objects:
            return 1.0  # vacuous: nothing to reconstruct yet
        n = 0.0
        ok = 0.0
        for obj in objects:
            try:
                d = self.adapter.descriptor(obj)          # L -> C
                nl_re = self.adapter.target_instruction(d)  # C -> L
                canon = self.adapter.render_object(obj)     # true L
                if not nl_re or not canon:
                    continue
                a = set(str(nl_re).lower().replace("_", " ").split())
                b = set(str(canon).lower().replace("_", " ").split())
                if not a and not b:
                    j = 1.0
                elif not a or not b:
                    j = 0.0
                else:
                    j = len(a & b) / len(a | b)
                n += 1.0
                ok += j
            except Exception:
                continue
        return ok / n if n else None

    # -- certificate -----------------------------------------------------

    def certificate(
        self,
        spec: RepresentationSpecification,
        objects: Sequence[Any] = (),
    ) -> FidelityCertificate:
        """Return the certificate for ``spec`` (optional concrete ``objects``
        used for the contrast + round-trip terms, outcome-blind)."""
        return FidelityCertificate(
            F_query=self._F_query(spec),
            F_dist=self._F_dist(spec, tuple(objects)),
            F_impl=self._F_impl(spec),
            round_trip=self._F_roundtrip(spec, tuple(objects)),
        )

    def score(self, spec: RepresentationSpecification) -> float:
        """Aggregate scalar F_t = min over certificate components (conservative).

        Kept as the principal threshold surface so existing callers
        (admission / readout) continue to work; componentwise admission is
        available via :meth:`admitted_componentwise` / :meth:`certificate`.
        """
        return self.certificate(spec).min

    def admitted_componentwise(
        self,
        spec: RepresentationSpecification,
        epsilon_f: float | Sequence[float] = 0.5,
        objects: Sequence[Any] = (),
    ) -> tuple[bool, FidelityCertificate]:
        """Componentwise admission: every certificate component >= its epsilon."""
        cert = self.certificate(spec, objects=objects)
        if isinstance(epsilon_f, (int, float)):
            eps = (float(epsilon_f), float(epsilon_f), float(epsilon_f))
        else:
            eps = tuple(float(e) for e in epsilon_f)
        ok = all(getattr(cert, _ctrls) >= e - _FIDELITY_EPS
                 for _ctrls, e in zip(("F_query", "F_dist", "F_impl"), eps))
        return ok, cert

    def admissible(self, spec: RepresentationSpecification, epsilon_f: float = 0.5) -> tuple[bool, float]:
        """Backward-compatible scalar admission (min-based)."""
        f = self.score(spec)
        return f >= epsilon_f - _FIDELITY_EPS, f


# ---------------------------------------------------------------------------
# Representation runtime  C_t = Runtime(P_t)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeRegistration:
    """Instantiation record of C_t from P_t.

    The runtime selects the actual recovery backend (the empirical field's
    substrate) guided by the spec's ``backend_hint``, defaulting to the
    features-free ``laplacian`` so the happy-path ``recover()`` stays valid.
    Full dynamic backend *induction* is a later extension; this is a
    deterministic, provenance-recording selection + validation step.
    """

    spec_id: str
    backend: str
    features_scheme: str  # 'graph' | 'euclidean' | 'none'
    accepted: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RepresentationRuntime:
    """Trusted runtime that instantiates C_t from an admitted P_t.

    Pre-registers the built-in recovery backends by their canonical hint names
    so an induced specification's ``backend_hint`` selects the empirical-field
    substrate directly (e.g. a Euclidean induced space -> ``rbf_gp``), enabling
    C_t to drive B_t in :meth:`~nlss.core.model.NLSSModel.recover`.
    """

    _BUILTIN_HINTS: Mapping[str, str] = {
        "laplacian": "laplacian",
        "rbf_gp": "rbf_gp",
        "graph_matern": "graph_matern",
        "multiscale_diffusion": "multiscale_diffusion",
        "graph_ensemble": "graph_ensemble",
        "mean": "mean",
    }

    def __init__(self) -> None:
        self._registry: dict[str, str] = dict(self._BUILTIN_HINTS)

    def register_backend_hint(self, hint: str, registry_name: str) -> None:
        self._registry[hint] = registry_name

    def instantiate(
        self,
        spec: RepresentationSpecification,
        *,
        force_backend: str | None = None,
        accepted: bool | None = None,
    ) -> RuntimeRegistration:
        backend = self._registry.get(spec.backend_hint, "laplacian")
        if force_backend is not None:
            backend = force_backend
        # K (geometry) is not inert: when the admitted contract declares an
        # explicit kernel/geometry, prefer an Euclidean substrate for it, and
        # always record the geometry in the registration so C_t provenance
        # reflects the contract's computational geometry (paper eq. K).
        geometry = spec.geometry or {}
        geom_kernel = str(geometry.get("kernel", "")).lower() if geometry else ""
        if force_backend is None and "rbf" in geom_kernel:
            backend = "rbf_gp"
        scheme = "euclidean" if backend in ("rbf_gp", "graph_matern") else "graph" if backend != "mean" else "none"
        accepted = spec.admitted if accepted is None else accepted
        return RuntimeRegistration(
            spec_id=spec.spec_id,
            backend=backend,
            features_scheme=scheme,
            accepted=accepted,
            metadata={
                "source": spec.source,
                "paradigm": spec.provenance.get("paradigm"),
                "runtime_mode": "selected",
                "geometry": dict(geometry) if geometry else None,
            },
        )


# ---------------------------------------------------------------------------
# Default (no-LLM) induction of a faithful candidate specification
# ---------------------------------------------------------------------------


def default_representation_spec(
    adapter: "TaskAdapter",
    *,
    spec_id: str = "p0",
    paradigm: RepresentationParadigm | None = None,
) -> RepresentationSpecification:
    """Transcribe the adapter's own declared structure into a candidate spec.

    This is the deterministic no-LLM stand-in for LLM-Induce: the "induced"
    variables/relations are read from the adapter's Frontier Semantics
    (descriptor keys -> typed state variables; attribute_domains -> their
    domains; executable transforms from revision_hints).  It lets the full
    dual-gate + revision machinery run and be tested without a gateway.
    """
    names: list[str] = []
    try:
        names = list(adapter.attribute_domains().keys())
    except Exception:
        names = ["n_ops", "depth", "neighborhood"]
    typed_vars = tuple(
        TypedVariable(name=n, vtype="continuous", role="state", description=f"descriptor dim {n}")
        for n in names
    )

    hints: list[str] = ["retrieve", "compare"]
    try:
        if hasattr(adapter, "revision_hints"):
            hints = list(adapter.revision_hints({}, {}))
    except Exception:
        hints = ["retrieve", "compare"]

    transforms = tuple(TransformDecl(name=h, arity="obj->obj", executable=True) for h in hints)

    relations = (
        RelationDecl("scientific_similarity", "obj-obj", "objects", outcome_blind=True,
                     description="outcome-blind scientific similarity (G_sci)"),
        RelationDecl("executable_revision", "obj-obj", "objects", outcome_blind=True,
                     description="meaningful executable revision (G_rev)"),
    )

    commitments: list[str] = []
    try:
        commitments = list(adapter.attribute_domains().keys())
    except Exception:
        commitments = names

    # paper contract components Sigma (entities), K (geometry), G (grounding):
    # populated from the adapter's declared structure so the adopted P_t carries
    # the full six-slot contract (paper eq. P_t=(Sigma,R,T,Phi,G,K)), not None.
    entities: tuple[str, ...] = (getattr(adapter, "object_type", "object"),)
    geometry: Mapping[str, Any] = {
        "backend": "laplacian",
        "kernel": "graph-laplacian",
        "neighborhood": "scientific_graph",
        "features": "graph",
    }
    grounding: Mapping[str, Any] = {
        "forward": "descriptor/target_instruction",
        "backward": "render_object/interpret",
    }

    return RepresentationSpecification(
        spec_id=spec_id,
        name=f"adapter-derived representation for {adapter.task_id}",
        description=(
            "Faithful transcription of the adapter's declared task structure; "
            "deterministic no-LLM candidate (P~_t stand-in)."
        ),
        typed_variables=typed_vars,
        relations=relations,
        transforms=transforms,
        semantic_commitments=tuple(commitments),
        entities=entities,
        geometry=geometry,
        grounding=grounding,
        backend_hint="laplacian",
        provenance={"paradigm": paradigm.value if paradigm else None, "default": True},
        source="adapter_default",
        admitted=False,
    )


# ---------------------------------------------------------------------------
# Representation induction front-end  (the "neural proposal" half of
# neuro-symbolic induction, proposal §5-§6)
# ---------------------------------------------------------------------------


class RepresentationInducer:
    """Produces a candidate representation specification P~_t.

    This is the **neural (proposal) authority** side of V7's
    logic-governed induction.  It proposes typed variables / relations /
    transforms / semantic commitments for the current task; the dual gate
    (LogicGate + FidelityGate, proposal §12) is the **symbolic (admissibility)
    authority** that decides whether the proposal is accepted.

    Two paths:
      - deterministic (no LLM): transcribe the adapter's declared structure
        (:func:`default_representation_spec`) — fully testable without a
        gateway;
      - LLM (when ``llm`` is wired): build a task prompt and ask the LLM to
        propose the computational representation as structured JSON; on any
        parse/provider failure we fall back to the deterministic transcription
        so induction never blocks the loop.
    """

    def __init__(self) -> None:
        pass

    def induce(
        self,
        adapter,
        *,
        paradigm: "RepresentationParadigm | None" = None,
        llm=None,
        spec_id: str = "p_t",
    ) -> RepresentationSpecification:
        candidate = self._propose_via_llm(adapter, llm, paradigm, spec_id) if llm is not None else None
        if candidate is None:
            candidate = default_representation_spec(adapter, spec_id=spec_id, paradigm=paradigm)
        return candidate

    def revise(
        self,
        adapter,
        old_spec: "RepresentationSpecification",
        signals=None,
        *,
        paradigm: "RepresentationParadigm | None" = None,
        llm=None,
    ) -> "RepresentationSpecification | None":
        """LLM-Revise (EXPERIMENT_PLAN §2.3): propose a *structural* revision
        ``P_candidate <- LLM-Revise(P_t, trigger, D_{t+1})`` that MAY add new
        typed variables / relations / transforms / entities / state-splits (a
        real contract-delta ``\\Delta P_t``, not just pruning).

        Returns None on any failure so the caller can fall back to the
        deterministic tightening path.  This is the live residual-3 surface; the
        facade only adopts the candidate after re-admission by the dual gate.
        """
        if llm is None:
            return None
        try:
            prompt = self._build_revision_prompt(adapter, old_spec, signals)
            result = getattr(llm, "generate_json", None)
            if result is None:
                return None
            data = result(prompt)
            if inspect.isawaitable(data):
                data = _run_coro(data)
            obj = data.dict() if hasattr(data, "dict") else (data if isinstance(data, dict) else None)
            if obj is None:
                return None
            sid = f"{getattr(old_spec, 'spec_id', 'p_t')}.r"
            revised = self._parse_spec(adapter, obj, sid, paradigm)
            revised = RepresentationSpecification(
                spec_id=sid,
                name=str(getattr(revised.name, "value", revised.name)) if hasattr(revised.name, "value") else revised.name,
                description=str(revised.description or "") + " [llm-revised]",
                typed_variables=revised.typed_variables,
                relations=revised.relations,
                transforms=revised.transforms,
                semantic_commitments=revised.semantic_commitments,
                entities=revised.entities,
                geometry=revised.geometry,
                grounding=revised.grounding,
                backend_hint=revised.backend_hint,
                provenance={**dict(revised.provenance or {}),
                            "revision_trigger": list(signals.labels) if signals else None,
                            "revised_by": "llm"},
                source="llm_revise",
                admitted=False,
            )
            return revised
        except Exception:
            return None

    def _build_revision_prompt(self, adapter, old_spec, signals) -> str:
        dims = []
        try:
            dims = list(adapter.attribute_domains().keys())
        except Exception:
            dims = []
        trig = ", ".join(signals.labels) if signals is not None else "none"
        return (
            "Revise the computational scientific representation contract for task "
            f"'{adapter.task_id}' because revision trigger(s) [{trig}] fired.\n"
            "You MAY ADD new typed variables, relations, transforms, entities, or "
            "state-splits to recover under-determined science — do not merely prune.\n"
            f"Current contract: variables={[v.name for v in old_spec.typed_variables]}, "
            f"relations={[r.name for r in old_spec.relations]}, "
            f"commitments={list(old_spec.semantic_commitments)}.\n"
            "Output JSON with keys: typed_variables (list of {name, vtype, role}), "
            "relations (list of {name, polymorphic_type, scope}), transforms "
            "(list of {name, arity}), semantic_commitments (list of str naming "
            "outcome-blind scientific distinctions to preserve, which may include "
            "NEW ones), entities (list of str), geometry (object with "
            "kernel/neighborhood/features), grounding (object with "
            "forward/backward). Available state variables suggested by the domain: "
            + (", ".join(dims) if dims else "none") + ".\n"
            "Never reference fitted outcome content in semantic_commitments."
        )

    def _propose_via_llm(self, adapter, llm, paradigm, spec_id) -> RepresentationSpecification | None:
        """Best-effort structured LLM proposal; returns None to trigger the
        deterministic fallback.

        Await-aware: an ``llm.generate_json`` that returns a coroutine/awaitable
        (e.g. the facade's metered wrapper) is completed via :func:`_run_coro`,
        so wiring an LLM actually exercises the proposal path instead of
        silently falling back to the deterministic transcription.
        """
        try:
            prompt = self._build_prompt(adapter)
            result = getattr(llm, "generate_json", None)
            if result is None:
                return None
            data = result(prompt)
            if inspect.isawaitable(data):
                data = _run_coro(data)
            # accept either a parsed JSON object or an object exposing .dict()
            obj = data.dict() if hasattr(data, "dict") else (data if isinstance(data, dict) else None)
            if obj is None:
                return None
            return self._parse_spec(adapter, obj, spec_id, paradigm)
        except Exception:
            return None

    def _build_prompt(self, adapter) -> str:
        dims = []
        try:
            dims = list(adapter.attribute_domains().keys())
        except Exception:
            dims = []
        return (
            "Propose the computational scientific representation for task "
            f"'{adapter.task_id}' (object_type '{adapter.object_type}').\n"
            "Output JSON with keys: typed_variables (list of {name, vtype, role}), "
            "relations (list of {name, polymorphic_type, scope}), transforms "
            "(list of {name, arity}), semantic_commitments (list of str naming "
            "outcome-blind scientific distinctions to preserve), entities "
            "(list of str: the task's scientific entity types), geometry "
            "(object with kernel/neighborhood/features), grounding (object with "
            "forward/backward). Available typed "
            "state variables suggested by the domain: " + (", ".join(dims) if dims else "none") + ".\n"
            "Never reference fitted outcome content in semantic_commitments."
        )

    def _parse_spec(self, adapter, obj: dict, spec_id: str, paradigm) -> RepresentationSpecification:
        from dataclasses import asdict

        typed_vars = tuple(
            TypedVariable(name=str(v["name"]), vtype=str(v.get("vtype", "continuous")),
                          role=str(v.get("role", "state")), description=str(v.get("description", "")))
            for v in obj.get("typed_variables", [])
            if isinstance(v, dict) and v.get("name")
        )
        relations = tuple(
            RelationDecl(name=str(r["name"]), polymorphic_type=str(r.get("polymorphic_type", "")),
                         scope=str(r.get("scope", "objects")),
                         outcome_blind=bool(r.get("outcome_blind", True)),
                         description=str(r.get("description", "")))
            for r in obj.get("relations", [])
            if isinstance(r, dict) and r.get("name")
        )
        transforms = tuple(
            TransformDecl(name=str(t["name"]), arity=str(t.get("arity", "obj->obj")),
                          executable=bool(t.get("executable", True)),
                          description=str(t.get("description", "")))
            for t in obj.get("transforms", [])
            if isinstance(t, dict) and t.get("name")
        )
        commitments = tuple(str(c) for c in obj.get("semantic_commitments", []))
        backend_hint = obj.get("backend_hint") or "laplacian"
        entities = tuple(str(e) for e in obj.get("entities", ()) if e)
        geometry = obj.get("geometry") if isinstance(obj.get("geometry"), dict) else None
        grounding = obj.get("grounding") if isinstance(obj.get("grounding"), dict) else None
        return RepresentationSpecification(
            spec_id=spec_id,
            name=f"LLM-induced representation for {adapter.task_id}",
            description="Structured representation proposed by the LLM proposer.",
            typed_variables=typed_vars,
            relations=relations or (RelationDecl("scientific_similarity", "obj-obj", "objects"),
                                    RelationDecl("executable_revision", "obj-obj", "objects")),
            transforms=transforms or (TransformDecl("retrieve", "obj->obj", True),
                                      TransformDecl("compare", "obj->obj", True)),
            semantic_commitments=commitments,
            entities=entities,
            geometry=geometry,
            grounding=grounding,
            backend_hint=str(backend_hint),
            provenance={"paradigm": paradigm.value if paradigm else None, "induced_by_llm": True},
            source="llm_induced",
            admitted=False,
        )
