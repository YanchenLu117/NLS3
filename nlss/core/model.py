"""NLSSModel — the V7 three-operation facade (FROZEN contract, §3.1).

    nlss.update(evidence_batch)
    space_state = nlss.recover()
    readout     = nlss.readout(space_state, token_budget=...)

and the proposal §5 model triangle ``M_t = (C_t, Gamma_t, B_t)``:

    C_t     <- ``self.backend``                 (a RecoveryBackend, e.g. rbf_gp)
    Gamma_t <- :class:`LanguageCorrespondence`  (wraps adapter semantics, read-only)
    B_t     <- ``space_state.posterior``        (a PosteriorView from fit_posterior)

This module is purely additive.  It composes the frozen components (readout.py,
recovery/, evidence.py, interfaces.py) and rewrites none of them.
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TYPE_CHECKING

# ---------------------------------------------------------------------------
# eta / gamma separation (T3): two distinct thresholds, never aliased.
#
#   gamma (value-space cut)   : backend ``RecoveryInput.solution_threshold``,
#                               used as p_t = norm_cdf((mean - gamma)/std) to
#                               derive the *value* boundary of the field (BH
#                               top-5% yield, GB1 fitness, MADE stability cut).
#   eta   (probability/rank): the posterior-probability cutoff
#                               S_t(eta) = {h : p_t(h) >= eta} used by
#                               ``solution_set`` and the readout family views.
# ---------------------------------------------------------------------------

# Default eta: non-zero so S_t(eta) is a *high-probability* subset instead of
# silently the whole belief space when the posterior is uniform at 0.5.
DEFAULT_ETA: float = 0.5

# Smallest spread in solution_probability that counts as "discriminative".
# Below this the posterior carries no resolution (e.g. laplacian with no
# value observations => every node p=0.5) and S_t(eta) would otherwise report
# "all objects" as if it were a real solution set.
DISCRIMINATION_EPS: float = 1e-9

from .interfaces import TaskAdapter
from .readout import CoverageFrontierReadout, ReadoutPacket, SpaceReadout, render_readout
from .acquisition import (
    COV,
    OPEN,
    SOL,
    ExplorationController,
    ExplorationGear,
)
from .representation import (
    FidelityGate,
    LogicGate,
    LogicReport,
    RepresentationInducer,
    RepresentationParadigm,
    RepresentationRuntime,
    RepresentationSpecification,
    RuntimeRegistration,
)
from .revision import (
    RepresentationRevisioner,
    RevisionReport,
    RevisionSignal,
    RevisionTrigger,
)
from .state import SolutionSpaceState
from .types import (
    GroundedObject,
    LanguageHypothesis,
    Observation,
    RevisionGraph,
    ScientificGraph,
    VerificationResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..llm.base import LLMProvider
    from ..recovery.base import RecoveryBackend


# ---------------------------------------------------------------------------
# EvidenceLog: the immutable evidence record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One immutable evidence-log entry.

    ``kind`` is one of ``hypothesis | object | observation | validity``; exactly
    the matching payload field is set.  An ``object`` kind indicates a
    pre-grounded object (already canonicalized) inserted verbatim.
    """

    kind: str
    hypothesis: LanguageHypothesis | None = None
    object: GroundedObject | None = None
    observation: Observation | None = None
    validity: VerificationResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EvidenceLog:
    """Append-only, first-class evidence log.  Supports ``len()``, indexing and
    iteration, and exposes a ``snapshot()`` of the log contents for audit.

    Semantics: the log records RAW appends — re-grounding an already-present
    hypothesis or re-inserting a duplicate observation appends a new entry, so
    ``len(log)`` is a mutation counter, not a distinct-count metric.  Distinct
    scientific content lives in ``state.objects`` / ``state.observations``
    (keyed by id, hence idempotent).  A re-entrant lock makes appends safe
    even if the facade is driven from a worker pool.
    """

    __slots__ = ("_records", "_lock")

    def __init__(self, records: Sequence[EvidenceRecord] = ()) -> None:
        self._records: list[EvidenceRecord] = list(records)
        self._lock = threading.Lock()

    def append(self, record: EvidenceRecord) -> None:
        with self._lock:
            self._records.append(record)

    def extend(self, records: Sequence[EvidenceRecord]) -> None:
        with self._lock:
            self._records.extend(records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, index):
        return self._records[index]

    def snapshot(self) -> tuple[EvidenceRecord, ...]:
        with self._lock:
            return tuple(self._records)


# ---------------------------------------------------------------------------
# Gamma_t — the language-layer correspondence (read-only wrapper)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LanguageCorrespondence:
    """Gamma_t — the (now bidirectional) language correspondence.

    V7 Core proposal §10: the correspondence must be two-way.

    Forward   (Gamma^right):  language -> computable  — ``descriptor`` /
              ``target_instruction`` translate a hypothesis toward a
              computational specification.
    Backward  (Gamma^left):   computable -> language  — ``interpret_marginal`` /
              ``render_state_to_language`` turn a recovered computational state
              back into scientifically meaningful language.

    Wraps the adapter's FrontierSemantics ``descriptor`` / ``target_instruction``
    and ``render_object``; it never mutates the adapter.
    """

    adapter: TaskAdapter

    def descriptor(self, obj: GroundedObject) -> Mapping[str, Any]:
        return self.adapter.descriptor(obj)

    def target_instruction(self, descriptor: Mapping[str, Any]) -> str:
        return self.adapter.target_instruction(descriptor)

    def render_object(self, obj: GroundedObject) -> str:
        return self.adapter.render_object(obj)

    # -- backward: computable -> language (Gamma_t^left) ------------------

    def interpret_marginal(self, obj: GroundedObject, marginal=None) -> str:
        """Render one computational object + its empirical belief into language.

        This is the minimal backward primitive: it names the object and, when a
        posterior marginal is supplied, its mean / uncertainty / solution
        probability in words — so a recovered computational solution becomes a
        natural-language scientific candidate again (Gamma^left).
        """
        head = self.render_object(obj)
        if marginal is None:
            return head
        words = [head]
        if getattr(marginal, "mean", None) is not None:
            words.append(f"empirical mean {float(marginal.mean):.4g}")
        if getattr(marginal, "std", None) is not None:
            words.append(f"uncertainty {float(marginal.std):.4g}")
        if getattr(marginal, "solution_probability", None) is not None:
            words.append(f"solution probability {float(marginal.solution_probability):.3f}")
        return " — ".join(words)

    def render_state_to_language(
        self,
        state: "SolutionSpaceState",
        *,
        solution_ids: Sequence[str] | None = None,
        closed_under: str = "supported",
    ) -> str:
        """Gamma^left over a recovered set: map computational solution structure
        back to a natural-language solution-space readout.

        Messages are compressed under a ``closed_under`` label so the recovered
        set is reported as *supported / probable-untested / under-covered*
        language views (proposal §18) rather than raw nodes.
        """
        lines: list[str] = []
        lines.append(f"## Recovered solution space ({closed_under})")
        if not solution_ids:
            lines.append("(no recovered solutions selected)")
            return "\n".join(lines)
        for oid in solution_ids:
            obj = state.objects.get(oid)
            if obj is None:
                continue
            marginal = None
            if state.posterior is not None:
                try:
                    marginal = state.posterior.marginal(oid)
                except Exception:
                    marginal = None
            lines.append(f"- {self.interpret_marginal(obj, marginal)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reusable grounding helper (compile -> verify -> canonicalize)
# ---------------------------------------------------------------------------


async def _ground_one(
    adapter: TaskAdapter,
    hypothesis: LanguageHypothesis,
    llm: "LLMProvider | None",
) -> GroundedObject | None:
    """Ground a single hypothesis (Compile -> Verify -> Canonicalize), reusing
    the same semantics as ``NLSSLoop.ground``."""
    raw = await adapter.compile(hypothesis, llm)
    verdict = adapter.verify(raw)
    if not verdict.valid:
        if verdict.repaired_object is None:
            return None
        raw = verdict.repaired_object
        verdict = adapter.verify(raw)
        if not verdict.valid:
            return None
    return adapter.canonicalize(raw)


_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_thread: threading.Thread | None = None


def _worker() -> asyncio.AbstractEventLoop:
    """A single persistent background event loop for driving async adapter
    calls from a sync facade when the caller is already inside a running loop."""
    global _worker_loop, _worker_thread
    if _worker_loop is None or not _worker_loop.is_running():
        _worker_loop = asyncio.new_event_loop()
        _worker_thread = threading.Thread(
            target=_worker_loop.run_forever, name="nlss-facade", daemon=True
        )
        _worker_thread.start()
    return _worker_loop


def _guard_empty_hypothesis(record: EvidenceRecord) -> bool:
    """Shared None-value guard (P1): ``update`` and ``aupdate`` must behave
    identically when a hypothesis record carries no payload.  Warn and return
    True (meaning "skip this record") when ``record.hypothesis is None``."""
    if record.hypothesis is None:
        warnings.warn(
            "EvidenceRecord(kind='hypothesis') with hypothesis=None skipped",
            RuntimeWarning,
        )
        return True
    return False



def _run(coro):
    """Run an async coroutine to completion from the synchronous facade.

    - No running loop: drive it with ``asyncio.run``.
    - Already inside a running loop: ship it to the persistent worker loop and
      block for the result (``run_coroutine_threadsafe``), so calling the sync
      facade from within a benchmark event loop does not crash
      ("event loop is already running").
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result = asyncio.run_coroutine_threadsafe(coro, _worker())
    return result.result()


class _MeteredLLM:
    """Metering wrapper that records every real LLM call into the §4.3 ledger.

    Wraps a V6 :class:`LLMProvider` and forwards its public surface
    (``provider_name`` / ``model_name`` / ``generate`` / ``generate_json``).
    The facade stores the wrapped instance as ``model.llm``, so any code that
    drives the provider through ``adapter.compile`` is automatically metered:
    each successful :meth:`generate` folds ``LLMUsage`` tokens in and counts
    one real LLM ``calls`` entry (T3 — ``calls`` is a real-call counter, not a
    facade bookkeeping entry).

    A V7 :class:`LLMBackend` (whose ``generate`` returns a ``GenerationResult``)
    is wrapped too if passed directly; the facade records its token metering.
    """

    __slots__ = ("_inner", "_model")

    def __init__(self, inner, model: "NLSSModel") -> None:
        self._inner = inner
        self._model = model

    @property
    def provider_name(self):
        return getattr(self._inner, "provider_name", None)

    @property
    def model_name(self):
        return getattr(self._inner, "model_name", None)

    async def generate(self, request) -> Any:
        result = await self._inner.generate(request)
        self._meter(result)
        return result

    async def generate_json(self, request=None, **kwargs) -> Any:
        result = await self._inner.generate_json(request, **kwargs)
        self._meter(result)
        return result

    @property
    def inner(self):
        """The underlying provider (escape hatch)."""
        return self._inner

    # -- metering -----------------------------------------------------------

    def _meter(self, result: Any) -> None:
        """Fold whichever usage shape the result carries into the ledger."""
        # V7 GenerationResult: has input_tokens / output_tokens attributes.
        if getattr(result, "input_tokens", None) is not None:
            self._model.record_generation(result)
            return
        # V6 LLMResponse: usage is on ``.usage``.
        usage = getattr(result, "usage", None)
        if usage is not None:
            inp = int(getattr(usage, "input_tokens", 0) or 0)
            out = int(getattr(usage, "output_tokens", 0) or 0)
            self._model.record_usage(token_input=inp, token_output=out, calls=1)


# ---------------------------------------------------------------------------
# NLSSModel — the V7 facade
# ---------------------------------------------------------------------------


class NLSSModel:
    """Three-operation facade over the frozen core components.

    Parameters
    ----------
    adapter:
        The benchmark TaskAdapter (also provides FrontierSemantics).
    backend:
        A RecoveryBackend (C_t); default None -> ``laplacian`` from the registry
        (features-free, so the happy-path ``recover()`` genuinely fits).
    readout:
        A SpaceReadout; defaults to :class:`CoverageFrontierReadout`.
    llm:
        Optional LLMProvider for hypotheses that need LLM compilation.
    solution_threshold:
        The **value-space cut gamma** of section 5.3: the backend threshold used
        to derive ``p_t = norm_cdf((mean - gamma)/std)`` for each node (BH top-5%
        yield, GB1 fitness, MADE stability cut).  It is the ``gamma`` of the
        decoupled eta/gamma pair (T3) and is recorded in
        ``space_state.metadata["value_threshold_gamma"]``.  Default 0.0 keeps
        ``marginal().solution_probability`` materialized even on features-free
        backends; benchmarks with a domain value cut pass their own gamma.
    eta:
        The **probability cutoff** (T3) used for the posterior rank set
        ``S_t(eta) = {h : p_t(h) >= eta}`` by :meth:`solution_set` and by the
        readout family views.  Non-zero by default (0.5, "high-probability") so
        a uniform posterior does not silently report the whole belief space as
        a "solution set".  Recorded in ``space_state.metadata["eta"]``.
    paradigm:
        The representation-paradigm arm (V7 Core §A.5 / Claim C4).  Defaults to
        ``V7_FULL`` (logic gate + fidelity gate active).  V7_NO_LOGIC disables
        the dual gate; V1/V6 use the fixed non-adaptive paths.
    epsilon_f:
        The fidelity threshold ``epsilon_F`` of the dual gate (proposal §11-§12).
        A representation is admitted only when ``F_t >= epsilon_f`` under the
        Full V7 paradigm.
    representation:
        Optionally pre-supply a :class:`RepresentationSpecification` (P_t).  If
        None, one is induced on first use from the adapter
        (:func:`default_representation_spec`) and admitted through the dual gate.
    """

    def __init__(
        self,
        adapter: TaskAdapter,
        backend: "RecoveryBackend | None" = None,
        readout: SpaceReadout | None = None,
        llm: "LLMProvider | None" = None,
        *,
        solution_threshold: float | None = 0.0,
        eta: float = DEFAULT_ETA,
        paradigm: RepresentationParadigm = RepresentationParadigm.V7_FULL,
        epsilon_f: float = 0.5,
        representation: RepresentationSpecification | None = None,
        exploration: ExplorationGear = ExplorationGear.OFF,
    ) -> None:
        self.adapter = adapter
        self.backend = backend
        self._explicit_backend = backend  # user-pinned; None => representation-led
        self.llm = llm
        # gamma (value-space cut) -> the backend; never aliased with eta.
        self.solution_threshold = solution_threshold
        # eta (probability/rank cutoff) -> solution_set / readout family views.
        self.eta = eta
        # V7 representation layer (proposal §5, §7, §11-§12, §21-§25).
        self.paradigm = paradigm
        self.epsilon_f = epsilon_f
        self.logic = LogicGate(adapter)
        self.fidelity = FidelityGate(adapter)
        self.runtime = RepresentationRuntime()
        self.revisioner = RepresentationRevisioner(adapter, epsilon_f=epsilon_f)
        self._inducer = RepresentationInducer()
        # exploration dial (acquisition layer, proposal §17 / v0.1)
        self.exploration = exploration
        self.exploration_controller = ExplorationController()
        # P_t: lazily induced/admitted on first access (see _ensure_representation).
        self._representation: RepresentationSpecification | None = representation
        self._runtime_registration: RuntimeRegistration | None = None
        self._representation_resolved: bool = False
        # section 4.3 token ledger -- C_in / C_out / C_total.  ``calls`` counts
        # ONLY real LLM calls (recorded via the metering LLM wrapper or
        # record_usage); facade operations never inflate it (T3).
        self._token_ledger: dict[str, int] = {"input": 0, "output": 0, "total": 0, "calls": 0}
        self._ledger_lock = threading.Lock()
        # NB: named ``reader`` (not ``readout``) so the ``readout`` method
        # stays callable as the V7 contract operation.
        self.reader = readout if readout is not None else CoverageFrontierReadout()
        # Wrap the LLM so every actual provider call auto-records token usage
        # into the §4.3 ledger (T3 wiring: the single real-LLM accounting point).
        if llm is not None:
            self.llm = _MeteredLLM(llm, self)

        self.state = SolutionSpaceState(
            task_id=adapter.task_id,
            round_index=0,
            objects={},
            hypotheses={},
            scientific_graph=ScientificGraph(node_ids=(), edges=()),
            revision_graph=RevisionGraph(node_ids=(), edges=()),
            observations={},
        )
        self.evidence_log = EvidenceLog()

        if backend is None:
            from ..recovery.registry import create_backend

            # P0-1b: the default backend is the features-free graph ``laplacian``
            # so the happy path ``recover()`` genuinely fits instead of silently
            # warning-and-degrading on every call.  A features-requiring backend
            # (e.g. rbf_gp) is only used when explicitly configured with features.
            self.backend = create_backend("laplacian")

    # -- update -----------------------------------------------------------

    def update(self, evidence_batch: Sequence[EvidenceRecord]) -> dict[str, Any]:
        """Process one evidence batch; returns a lightweight snapshot.

        NB: mutates ``state.objects/observations/hypotheses`` on the calling
        thread.  These dict writes are not lock-protected (only the
        :class:`EvidenceLog` is); the facade is expected to be driven from a
        single thread (or externally synchronized), as is the benchmark loop.
        """
        state = self.state
        for record in evidence_batch:
            kind = record.kind
            if kind == "hypothesis":
                if _guard_empty_hypothesis(record):
                    self.evidence_log.append(record)
                    continue
                obj = _run(_ground_one(self.adapter, record.hypothesis, self.llm))
                if obj is not None:
                    # objects keyed by content-derived object_id: two different
                    # hypothesis_ids that ground to the same object collapse to
                    # one node; the hypotheses map is last-writer-wins by design.
                    state.objects[obj.object_id] = obj
                    state.hypotheses[obj.object_id] = record.hypothesis
                else:
                    # ungroundable hypothesis — RevisionTrigger.UNGROUNDABLE
                    state.metadata["ungroundable_hypotheses"] = int(
                        state.metadata.get("ungroundable_hypotheses", 0) or 0
                    ) + 1
            elif kind == "object":
                if record.object is not None:
                    state.objects[record.object.object_id] = record.object
            elif kind == "observation":
                if record.observation is not None:
                    state.observations[record.observation.object_id] = record.observation
            elif kind == "validity":
                pass  # verdict only, carried in the log
            else:
                raise ValueError(f"unknown evidence kind: {kind!r}")
            self.evidence_log.append(record)
        return self.snapshot()

    async def aupdate(self, evidence_batch: Sequence[EvidenceRecord]) -> dict[str, Any]:
        """Async variant of :meth:`update` for callers already inside an event
        loop (avoids the background-thread hop that :meth:`update` uses when a
        loop is running)."""
        state = self.state
        for record in evidence_batch:
            kind = record.kind
            if kind == "hypothesis":
                if _guard_empty_hypothesis(record):
                    self.evidence_log.append(record)
                    continue
                obj = await _ground_one(self.adapter, record.hypothesis, self.llm)
                if obj is not None:
                    state.objects[obj.object_id] = obj
                    state.hypotheses[obj.object_id] = record.hypothesis
                else:
                    state.metadata["ungroundable_hypotheses"] = int(
                        state.metadata.get("ungroundable_hypotheses", 0) or 0
                    ) + 1
            elif kind == "object":
                if record.object is not None:
                    state.objects[record.object.object_id] = record.object
            elif kind == "observation":
                if record.observation is not None:
                    state.observations[record.observation.object_id] = record.observation
            elif kind == "validity":
                pass
            else:
                raise ValueError(f"unknown evidence kind: {kind!r}")
            self.evidence_log.append(record)
        return self.snapshot()

    # -- recover ----------------------------------------------------------

    def recover(self, features: Mapping[str, Any] | None = None) -> SolutionSpaceState:
        """Rebuild the graphs, fit the posterior (B_t), and return space_state.

        ``features`` is an optional outcome-blind object_id -> feature-array map
        required by Euclidean backends (e.g. ``rbf_gp``); graph backends ignore it.
        If no features are supplied, the explicit :meth:`TaskAdapter.feature_vectors`
        contract (P0-1b) is consulted and outcome-blind features are derived
        automatically when the adapter provides them (e.g. MADE).  The default
        backend is the features-free ``laplacian`` so the happy-path
        ``space_state = nlss.recover()`` genuinely fits; a features-requiring
        backend is used only when explicitly configured with features.  The
        backend actually used is recorded in ``state.metadata["recovery_backend"]``.
        """
        state = self.state
        objects = tuple(state.objects.values())
        state.scientific_graph = self.adapter.build_scientific_graph(objects)
        state.revision_graph = self.adapter.build_revision_graph(objects)

        # Explicit features contract on TaskAdapter (P0-1b): we call the method
        # declared on the interface -- never duck-type via hasattr -- so the
        # features contract is discoverable and the default returns None for
        # adapters that do not provide outcome-blind features.
        if features is None:
            try:
                features = self.adapter.feature_vectors(objects)
            except Exception:
                features = None

        from ..recovery.base import RecoveryInput

        recovery_input = RecoveryInput(
            objects=objects,
            scientific_graph=state.scientific_graph,
            observations=tuple(state.observations.values()),
            features=features,
            solution_threshold=self.solution_threshold,
        )

        used_backend_name, used_posterior = self._fit_posterior(recovery_input)
        state.posterior = used_posterior
        state.metadata["recovery_backend"] = used_backend_name
        # gamma (value-space cut) and eta (probability/rank cut) are stored
        # separately (T3) so consumers never alias the two.
        state.metadata["value_threshold_gamma"] = self.solution_threshold
        state.metadata["eta"] = self.eta
        # Mark whether the fitted posterior actually separates nodes (T1/T3):
        # a uniform/everything-equal posterior (e.g. laplacian, no value
        # observations => every node p=0.5) is NON-discriminative and must not
        # be mistaken for a real solution set.
        state.metadata["posterior_discriminative"] = self._posterior_discriminative(
            used_posterior, tuple(state.objects)
        )
        return state

    def _posterior_discriminative(self, posterior, objects) -> bool:
        """True iff the posterior's solution probabilities actually separate the
        objects (spread above :data:`DISCRIMINATION_EPS`).

        With no value observations and no features (e.g. the laplacian default),
        every node has the same solution_probability (0.5) — there is no signal,
        so we return False and the facade flags it rather than silently reporting
        "all objects" as a solution set.
        """
        probs: list[float] = []
        for oid in objects:
            try:
                m = posterior.marginal(oid)
            except Exception:
                m = None
            if m is not None and m.solution_probability is not None:
                probs.append(float(m.solution_probability))
        if len(probs) < 2:
            return False
        return (max(probs) - min(probs)) > DISCRIMINATION_EPS

    def _fit_posterior(self, recovery_input) -> tuple[str, Any]:
        """Fit the configured backend with graceful degradation.

        Returns ``(backend_name, posterior)``.  The configured backend
        (C_t, e.g. ``rbf_gp``) is used when it can fit the given input; if it
        cannot (features absent/incomplete, empty graph, or a fit error), we
        fall back to a features-free graph backend (``laplacian``), then
        ``mean`` for an empty graph — so the V7 happy-path ``recover()`` never
        crashes.  A ``RuntimeWarning`` is emitted whenever a fallback replaces
        the configured backend, so a silently downgraded backend is never
        invisible.  Callers that need the *exact* backend and its errors can
        drive ``backend.fit_posterior(...)`` directly.
        """
        from ..recovery.registry import create_backend

        try:
            posterior = self._backend().fit_posterior(recovery_input)
            return self.backend.name, posterior
        except Exception as exc:
            warnings.warn(
                "rec backend '%s' failed to fit (%s: %s); degrading to a "
                "graph/mean backend. See state.metadata['recovery_backend']."
                % (getattr(self.backend, "name", "?"), type(exc).__name__, exc),
                RuntimeWarning,
            )
        try:  # laplacian: graph-only, works without features
            fallback = create_backend("laplacian")
            return fallback.name, fallback.fit_posterior(recovery_input)
        except Exception:
            fallback = create_backend("mean")  # works even on an empty graph
            return fallback.name, fallback.fit_posterior(recovery_input)

    # -- readout ----------------------------------------------------------

    def readout(
        self,
        space_state: SolutionSpaceState,
        token_budget: int = 1800,
    ) -> str:
        """One step from a fitted space_state to token-budgeted text.

        ``token_budget`` must be a positive int.  NB: the frozen
        :func:`render_readout` enforces the budget by a documented
        4-characters-per-token approximation (``text[: budget * 4]``); it is a
        character cap, not a tokenizer-verified token cap.
        """
        if not isinstance(token_budget, int) or token_budget < 1:
            raise ValueError(f"token_budget must be a positive int, got {token_budget!r}")
        packet = self.readout_packet(space_state)
        return render_readout(packet, space_state, self.adapter, token_budget)

    def readout_packet(self, space_state: SolutionSpaceState) -> ReadoutPacket:
        """Convenience: return the raw ReadoutPacket (semantics = adapter).

        Routes through the single frozen formal readout path
        (:class:`CoverageFrontierReadout`) and feeds it the fitted posterior so
        the section-5.3 solution-family views are produced when available.  The
        legacy posterior-driven :class:`EvidenceBuilder` is deliberately NOT the
        facade's readout path (see core/evidence.py) and remains available only
        as a deprecated, explicit opt-in for backward compatibility.

        The posterior-driven family views are cut at the model's **eta** (the
        probability cutoff, T3) — distinct from the backend's ``gamma`` value
        cut — so ``supported`` / ``probable_untested`` are genuine
        high-probability families rather than "everything >= 0".
        """
        return self.reader.build_with_posterior(
            space_state, self.adapter, posterior=self.state.posterior, eta=self.eta
        )

    # -- V7 representation layer (P_t, L_tau, F_t, dual gate, revision) ------

    def _ensure_representation(self) -> None:
        """Induce and admit P_t on first use (lazy, paradigm-respecting)."""
        if self._representation_resolved:
            return
        self._representation_resolved = True
        if self._representation is None:
            self._representation = self.induce_representation()
        else:
            self._admit_reuse(self._representation)

    def induce_representation(self) -> RepresentationSpecification:
        """``widetilde P_t <- LLM-Induce(tau, H_le t, D_t)``  (proposal §5).

        Routes through the :class:`RepresentationInducer` (the neural proposal
        side of neuro-symbolic induction).  With an ``llm`` wired, the LLM
        proposes the computational representation as structured JSON; otherwise
        (and on any proposal failure) it falls back to a faithful deterministic
        transcription of the adapter's declared structure.  Either way the
        candidate then passes the dual gate (logic + fidelity) via
        :meth:`_admit_reuse`.
        """
        spec = self._inducer.induce(
            self.adapter,
            paradigm=self.paradigm,
            llm=self.llm,
            spec_id="p_t",
        )
        return self._admit_reuse(spec)

    def _admit_reuse(self, spec: RepresentationSpecification) -> RepresentationSpecification:
        """Re-run the dual gate (logic + fidelity) on a candidate P_t and, on
        admission, instantiate C_t via the runtime.  Records everything on the
        persistent state's metadata.
        """
        admitted_spec, logic_report, f = self.admit_representation(spec)
        self._representation = admitted_spec
        reg = self.runtime.instantiate(admitted_spec)
        self._runtime_registration = reg
        self.state.metadata["representation_spec_id"] = admitted_spec.spec_id
        self.state.metadata["representation_paradigm"] = self.paradigm.value
        self.state.metadata["representation_admitted"] = admitted_spec.admitted
        self.state.metadata["representation_backend"] = reg.backend
        self.state.metadata["representation_fidelity"] = f
        self.state.metadata["representation_logic_valid"] = logic_report.valid
        self.state.metadata["representation_fidelity_gate"] = self.paradigm.runs_fidelity_gate
        self.state.metadata["representation_logic_gate"] = self.paradigm.runs_logic_gate
        # store the three-component fidelity certificate (F_query, F_dist, F_impl)
        try:
            cert = self.fidelity.certificate(
                admitted_spec, objects=tuple(self.state.objects.values())
            )
            self.state.metadata["representation_fidelity_cert"] = {
                "F_query": cert.F_query, "F_dist": cert.F_dist, "F_impl": cert.F_impl,
                "round_trip_fidelity": cert.round_trip,
            }
        except Exception as exc:  # honest: never silently swallow the certificate
            self.state.metadata["representation_fidelity_cert"] = {"error": str(exc)}
            import warnings

            warnings.warn(f"fidelity certificate unavailable: {exc}", RuntimeWarning)
        return admitted_spec

    def admit_representation(
        self,
        spec: RepresentationSpecification,
    ) -> tuple[RepresentationSpecification, LogicReport, float]:
        """The dual gate (proposal §12): ``LogicValid(P_t; L_tau)`` AND
        ``F(P_t, H) >= epsilon_f``.  Full V7 enforces both; V7_NO_LOGIC enforces
        neither contract (the "value of logic" ablation arm); V1/V6 are
        non-adaptive and simply recorded.
        """
        logic_report = LogicReport(valid=True, checked=("bypass",))
        if self.paradigm.runs_logic_gate:
            logic_report = self.logic.check(spec)
        f = self.fidelity.score(spec)
        fid_ok = f >= self.epsilon_f if self.paradigm.runs_fidelity_gate else True
        admitted = bool(logic_report.valid and fid_ok)
        admitted_spec = RepresentationSpecification(
            spec_id=spec.spec_id,
            name=spec.name,
            description=spec.description,
            typed_variables=spec.typed_variables,
            relations=spec.relations,
            transforms=spec.transforms,
            semantic_commitments=spec.semantic_commitments,
            entities=spec.entities,          # Sigma (paper eq. P_t=(Sigma,R,T,Phi,G,K))
            geometry=spec.geometry,          # K
            grounding=spec.grounding,        # G
            backend_hint=spec.backend_hint,
            provenance=dict(spec.provenance),
            source=spec.source,
            admitted=admitted,
        )
        return admitted_spec, logic_report, float(f)

    @property
    def representation(self) -> RepresentationSpecification:
        """P_t — the current adopted representation specification (lazy)."""
        self._ensure_representation()
        assert self._representation is not None
        return self._representation

    @property
    def runtime_registration(self) -> RuntimeRegistration | None:
        self._ensure_representation()
        return self._runtime_registration

    def fidelity_score(self) -> float:
        """F_t(H_NL, C_t, Gamma_t) — the compact fidelity auditor (proposal §11)."""
        return self.fidelity.score(self.representation)

    def fidelity_certificate(self):
        """The three-component outcome-blind certificate (F_query, F_dist, F_impl),
        componentwise admission surface (paper §4.1)."""
        return self.fidelity.certificate(
            self.representation, objects=tuple(self.state.objects.values())
        )

    def representational_state(self) -> dict[str, Any]:
        """The completed persistent state  S^{V7} = (D_t, P_t, C_t, Gamma_t, B_t)
        as a summary (proposal §27).
        """
        self._ensure_representation()
        gamma = self.language_correspondence
        return {
            "D_t": {"evidence_count": len(self.evidence_log),
                    "n_observed": len(self.state.observations)},
            "P_t": {"spec_id": self.representation.spec_id,
                    "admitted": self.representation.admitted,
                    "semantic_commitments": self.representation.semantic_commitments},
            "C_t": {"backend": self._runtime_registration.backend if self._runtime_registration else None,
                    "features_scheme": self._runtime_registration.features_scheme if self._runtime_registration else None},
            "Gamma_t": {"bidirectional": True, "forward": "descriptor/target_instruction",
                        "backward": "interpret_marginal/render_state_to_language"},
            "B_t": {"posterior_fitted": self.state.posterior is not None,
                    "discriminative": bool(self.state.metadata.get("posterior_discriminative", False))},
        }

    def interpret_solution_space(
        self,
        space_state: SolutionSpaceState | None = None,
        *,
        eta: float | None = None,
        closed_under: str = "supported",
    ) -> str:
        """Gamma^left over the recovered computational space (proposal §18, §10).

        Converts the recovered ``S_{C,t}(eta)`` into a natural-language
        solution-space readout ``S_{NL,t}`` for the next reasoning step.
        """
        state = space_state if space_state is not None else self.state
        if state.posterior is None:
            return self.language_correspondence.render_state_to_language(
                state, solution_ids=(), closed_under=closed_under
            )
        # a uniform/non-discriminative posterior is not a recovered solution
        # space (T1/T3): never surface "everything" as the recovery.
        if not self._posterior_discriminative(state.posterior, tuple(state.objects)):
            return self.language_correspondence.render_state_to_language(
                state, solution_ids=(), closed_under=closed_under
            )
        sol_ids = self.solution_set(state, eta=eta)
        return self.language_correspondence.render_state_to_language(
            state, solution_ids=sol_ids, closed_under=closed_under
        )

    # -- evidence-driven representation revision (proposal §21-§25) ----------

    def check_revision(self) -> RevisionSignal:
        """Scan the four triggers (proposal §22).  Never fires for V1/V6."""
        return self.revisioner.scan(self.state, self.representation, paradigm=self.paradigm)

    def revise_representation(self, *, missing_data_policy: str = "exclude",
                              llm_inductor=None) -> RevisionReport:
        """On a fired trigger, propose + re-admit a revised P_t.

        Invariant ``D_{t+1} superset D_t`` is preserved: evidence is never
        touched; only the specification (and the field that will be rebuilt from
        it) changes (proposal §24).  A revised spec that FAILS the dual gate is
        NOT adopted: the old representation is kept and the field is never
        rebuilt under a rejected contract (so ``P`` and ``B_t`` stay consistent).
        ``missing_data_policy`` (retain/exclude/query) selects the re-grounding
        missing-data semantics.  Returns the :class:`RevisionReport`.
        """
        report = self.revisioner.decide_and_revise(
            self.state, self.representation, paradigm=self.paradigm,
            epsilon_f=self.epsilon_f, llm_inductor=llm_inductor
        )
        if report.needed and report.new_spec is not None and report.new_spec.admitted:
            self._representation = self._admit_reuse(report.new_spec)
            if self.state.objects:
                # invariant: evidence D_t unchanged, field B_t rebuilt under the
                # new C_t (proposal §24 — reinterpret old evidence, don't erase).
                self.recover()
                self.state.metadata["representation_field_rebuilt"] = True
            self._reground_evidence(missing_data_policy=missing_data_policy)
        elif report.needed:
            # rejected revision: keep the current (old) representation untouched.
            self.state.metadata["representation_field_rebuilt"] = False
            self.state.metadata["revision_rejected_kept_old"] = report.new_spec.spec_id if report.new_spec else None
        self.state.metadata["last_revision"] = {
            "needed": report.needed,
            "triggers": report.triggers,
            "adhered_logically": report.adhered_logically,
        }
        return report

    async def arevise_representation(self, *, missing_data_policy: str = "exclude",
                                     llm_inductor=None) -> RevisionReport:
        """Async variant of :meth:`revise_representation` that runs the
        re-grounding pass in this event loop (so re-grounding actually executes
        inside the Algorithm-1 agent loop rather than being skipped).  A revised
        spec that fails the dual gate is NOT adopted; the old P is kept and B_t
        is never rebuilt under a rejected contract."""
        report = self.revisioner.decide_and_revise(
            self.state, self.representation, paradigm=self.paradigm,
            epsilon_f=self.epsilon_f, llm_inductor=llm_inductor
        )
        if report.needed and report.new_spec is not None and report.new_spec.admitted:
            self._representation = self._admit_reuse(report.new_spec)
            if self.state.objects:
                self.recover()
                self.state.metadata["representation_field_rebuilt"] = True
            try:
                from .reground import ReGroundingPass, apply_missing_data_policy

                rg = await ReGroundingPass(self.adapter).re_ground_ledger(
                    self.state, self._representation
                )
                self._store_reground(rg, apply_missing_data_policy(rg, missing_data_policy),
                                     mode="executed_async", policy=missing_data_policy)
            except Exception as exc:  # pragma: no cover - defensive
                self.state.metadata["representation_reground"] = {"mode": "error", "error": str(exc)}
        elif report.needed:
            self.state.metadata["representation_field_rebuilt"] = False
            self.state.metadata["revision_rejected_kept_old"] = report.new_spec.spec_id if report.new_spec else None
        self.state.metadata["last_revision"] = {
            "needed": report.needed,
            "triggers": report.triggers,
            "adhered_logically": report.adhered_logically,
        }
        return report

    def _store_reground(self, rg, usable_ids, mode: str, policy: str = "exclude") -> None:
        """Record a re-grounding report onto the persistent state metadata."""
        self.state.metadata["representation_reground"] = {
            "regrounded": list(rg.regrounded),
            "excluded": list(rg.excluded),
            "queried": list(rg.queried),
            "unknown_attrs": [
                {"record_id": u.record_id, "attr": u.attr, "reason": u.reason}
                for u in rg.unknown_attrs
            ],
            "usable_ids": list(usable_ids),
            "mode": mode,
            "policy": policy,
        }

    def _reground_evidence(self, *, missing_data_policy: str = "exclude") -> None:
        """Best-effort re-grounding pass under the new contract (paper §4.3).

        Re-grounds every prior evidence record under the adopted P_t, records
        recoverable/unrecoverable attributes and missing-data policy impact.
        Runs only when no event loop is active (re-grounding's compile is
        async); otherwise it is recorded as skipped so the loop never blocks.
        """
        try:
            from .reground import ReGroundingPass, apply_missing_data_policy

            rg = ReGroundingPass(self.adapter).re_ground_ledger_sync(
                self.state, self._representation
            )
            self._store_reground(rg, apply_missing_data_policy(rg, missing_data_policy),
                                 mode="executed", policy=missing_data_policy)
        except TypeError:
            self.state.metadata["representation_reground"] = {"mode": "skipped_active_loop"}
        except Exception as exc:  # pragma: no cover - defensive
            self.state.metadata["representation_reground"] = {"mode": "error", "error": str(exc)}

    # -- exploration dial (acquisition layer, v0.1 gear + channel policies) ---

    def select_channel(self, rng=None) -> str:
        """Sample a proposal channel from the current gear's distribution."""
        if rng is None:
            import random

            rng = random.Random()
        return self.exploration_controller.select_channel(rng, self.exploration)

    def acquire(
        self,
        k: int = 1,
        *,
        rng=None,
        eta: float | None = None,
    ) -> list[tuple[str, str | None]]:
        """Propose ``k`` next actions under the current exploration gear.

        Returns ``[(channel, oid_or_None), ...]``.  For SOL/BND/COV the object
        is the top candidate of that channel's normalized policy over the
        currently constructed space.  For OPEN no in-space object exists: the
        entry is ``("OPEN", None)``, signalling the language-side proposer to
        generate a hypothesis that may leave ``dom(Gamma_t)`` (see
        :meth:`open_directive`); an ungroundable result then becomes a
        representation challenge via the UNGROUNDABLE revision trigger.
        """
        if rng is None:
            import random

            rng = random.Random()
        object_ids = list(self.state.objects)
        eta = self.eta if eta is None else eta
        out: list[tuple[str, str | None]] = []
        for _ in range(k):
            channel = self.exploration_controller.select_channel(rng, self.exploration)
            if channel == OPEN:
                out.append((OPEN, None))
                continue
            if not object_ids:
                out.append((channel, None))
                continue
            ranked = self.exploration_controller.rank(
                channel, object_ids, self.state.posterior, eta
            )
            out.append((channel, ranked[0] if ranked else None))
        return out

    def open_directive(self) -> str:
        """Prompt fragment for the OPEN channel: asks the language proposer for
        a scientifically coherent direction that may exceed the current
        representation (v0.1 §Open-world policy)."""
        return (
            "OPEN proposal: suggest a scientifically coherent hypothesis that "
            "may go beyond the current computational representation (it need not "
            "fall in dom(Gamma_t)). If it cannot be grounded, it is a "
            "representation-revision challenge, not a fabricated field score."
        )

    def batch_plan(self, q: int, residual=None) -> tuple[dict[str, int], tuple]:
        """Delegate integer per-channel allocation for a batch of ``q``."""
        return self.exploration_controller.batch_plan(q, self.exploration, residual)

    def recommended_gear(self) -> ExplorationGear:
        """Non-overriding gear suggestion from lightweight state signals."""
        meta = {
            "pure_exploit": False,
            "representation_inadequate": bool(
                self.state.metadata.get("ungroundable_hypotheses", 0)
                or not self.state.metadata.get("representation_admitted", True)
            ),
            # coverage "stall" only counts once there is evidence but the field
            # still cannot discriminate (a fresh empty model is not "stalled").
            "coverage_stalled": bool(self.state.observations)
            and not bool(self.state.metadata.get("posterior_discriminative", False)),
        }
        return self.exploration_controller.recommend(meta)

    # -- helpers -----------------------------------------------------------

    def _backend(self) -> "RecoveryBackend":
        """Resolve the empirical-field substrate ``C_t`` for B_t.

        - A user-pinned backend (passed at construction) is always honored.
        - Otherwise the **representation drives the field**: the backend
          selected by :class:`RepresentationRuntime` from the adopted P_t
          (e.g. a Euclidean induced space -> ``rbf_gp``) becomes the substrate,
          so ``recover()`` genuinely fits the induced computable space rather
          than a fixed default.  Falls back to ``laplacian`` if unavailable.
        """
        if self._explicit_backend is not None:
            self.backend = self._explicit_backend
            return self.backend
        self._ensure_representation()
        from ..recovery.kernel_factory import resolve_kernel
        from ..recovery.registry import create_backend

        # the adopted contract's geometry drives the substrate (kernel-per-
        # representation, EXPERIMENT_PLAN §2.1 Layer II) — not a bare default.
        decision = resolve_kernel(self._representation)
        selected = decision.backend
        # a runtime-driven backend (also geometry-derived) may be more specific
        if self._runtime_registration is not None and self._runtime_registration.backend in (
                "rbf_gp", "graph_matern", "multiscale_diffusion", "graph_ensemble", "mean"):
            selected = self._runtime_registration.backend
        self.state.metadata["kernel_decision"] = {
            "backend": selected,
            "kernel": decision.kernel,
            "features_scheme": decision.features_scheme,
            "geometry_kind": decision.geometry_kind,
            "provenance": dict(decision.provenance),
        }
        cur = getattr(self.backend, "name", None)
        if cur is None or cur != selected:
            try:
                self.backend = create_backend(selected)
            except Exception:
                self.backend = create_backend("laplacian")
        return self.backend

    # -- section 4.3 token ledger ------------------------------------------

    def _account(self, *, token_input: int = 0, token_output: int = 0, calls: int = 0) -> None:
        """Accumulate one entry into the cumulative token ledger.  Thread-safe.

        ``calls`` counts **only real LLM calls** (T3): facade operations
        (update/recover/readout) never touch it, so the ledger distinguishes a
        genuine provider call from a no-LLM accounting bookkeeping entry.
        """
        with self._ledger_lock:
            self._token_ledger["input"] += int(token_input)
            self._token_ledger["output"] += int(token_output)
            self._token_ledger["total"] += int(token_input) + int(token_output)
            self._token_ledger["calls"] += int(calls)

    def record_usage(self, *, token_input: int = 0, token_output: int = 0, calls: int = 1) -> None:
        """Record one concrete LLM-call usage event (C_in / C_out) plus that
        single real call into the cumulative §4.3 ledger.

        This is the explicit real-LLM accounting hook: benchmarks that wire a
        provider through the adapter call it (or rely on the metering LLM
        wrapper in :meth:`update` / :meth:`aupdate`) after each live
        generation.  Pure-computation runs simply never call it, so
        :meth:`token_usage` honestly reports ``(0, 0, 0, 0)`` instead of
        inventing calls.
        """
        self._account(calls=calls, token_input=token_input, token_output=token_output)

    def record_generation(self, generation_result) -> None:
        """Fold a :class:`~nlss.llm.base.GenerationResult` into the ledger.

        Convenience for the V7 ``LLMBackend`` path: reads ``input_tokens`` /
        ``output_tokens`` from the generation and records one real LLM call.
        Used by callers that drive an ``LLMBackend.generate`` through the
        facade (and by the metering wrapper for V6 ``LLMProvider`` paths).
        """
        inp = int(getattr(generation_result, "input_tokens", 0) or 0)
        out = int(getattr(generation_result, "output_tokens", 0) or 0)
        self.record_usage(token_input=inp, token_output=out, calls=1)

    def token_usage(self) -> dict[str, int]:
        """Cumulative §4.3 accounting: ``{input, output, total, calls}``.

        ``total = input + output``; ``calls`` counts **real LLM calls only**
        (T3) — a pure-computation facade run returns ``(0, 0, 0, 0)``.
        """
        with self._ledger_lock:
            return dict(self._token_ledger)

    # -- section 5.3 eta solution set --------------------------------------

    def solution_set(
        self,
        space_state: SolutionSpaceState | None = None,
        eta: float | None = None,
        *,
        by_rank: bool = False,
    ) -> tuple[str, ...]:
        """S_t(eta) = { h : p_t(h) >= eta } from the fitted posterior marginals.

        ``eta`` defaults to the model's **eta** (the probability cutoff, T3 —
        NOT the backend's ``gamma`` value cut).  With ``by_rank=False`` (the
        default) ``eta`` is an absolute probability cutoff in ``[0,1]``: a
        default of :data:`DEFAULT_ETA` (0.5) yields the *high-probability*
        posterior set, never "everything >= 0".

        With ``by_rank=True`` (edge-adapted HypoSpace-style, no value domain)
        ``eta`` is interpreted as a **rank / quantile cutoff** in ``(0,1]``: the
        top ``eta`` fraction of objects ordered by ``solution_probability``
        (e.g. ``eta=0.5`` -> top half).  For a truly non-discriminative
        posterior the result is flagged via
        ``space_state.metadata["solution_set_discriminative"] = False`` rather
        than being silently presented as a real solution set.

        This is the **implicit** representation of §5.3 (allowed not to be
        enumerated here); no admissible-set enumeration is performed.  Returns
        an empty tuple when the posterior is not fitted or a marginal lacks
        ``solution_probability``.
        """
        state = space_state if space_state is not None else self.state
        if state.posterior is None:
            return ()
        eta = self.eta if eta is None else eta
        probs: list[tuple[str, float]] = []
        for oid in state.objects:
            m = state.posterior.marginal(oid)
            if m.solution_probability is not None:
                probs.append((oid, float(m.solution_probability)))
        # Explicit non-discrimination marker (T1/T3): a uniform posterior must
        # surface as such on the state, never masquerade as a solution set.
        disc = self._posterior_discriminative(state.posterior, tuple(state.objects))
        state.metadata["solution_set_discriminative"] = disc
        if by_rank:
            frac = float(eta)
            if not (0.0 < frac <= 1.0):
                raise ValueError("by_rank=True requires eta in (0, 1]")
            ranked = sorted(probs, key=lambda t: t[1], reverse=True)
            k = int(round(len(ranked) * frac))
            return tuple(oid for oid, _ in ranked[:k])
        out = [oid for oid, p in probs if p >= float(eta)]
        return tuple(out)

    def snapshot(self) -> dict[str, Any]:
        # Deliberately does NOT resolve/induce the representation: induction may
        # consume an (LLM) call, so it must be an explicit step, never a side
        # effect of a lightweight snapshot (keeps LLM accounting honest).
        return {
            "task_id": self.state.task_id,
            "round_index": self.state.round_index,
            "n_objects": len(self.state.objects),
            "n_observations": len(self.state.observations),
            "n_hypotheses": len(self.state.hypotheses),
            "posterior_fitted": self.state.posterior is not None,
            "evidence_count": len(self.evidence_log),
            "token_usage": self.token_usage(),
            "paradigm": self.paradigm.value,
            "exploration": self.exploration.value,
            "representation": {
                "spec_id": self.state.metadata.get("representation_spec_id"),
                "admitted": bool(self.state.metadata.get("representation_admitted", False)),
                "backend": self.state.metadata.get("representation_backend"),
            },
        }

    @property
    def language_correspondence(self) -> LanguageCorrespondence:
        return LanguageCorrespondence(adapter=self.adapter)
