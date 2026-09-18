"""NLSS V6.1 core — state, grounding, relations, recovery, evidence, readout.

V6.1 (governance-approved, limited reopen): adds the formal SpaceReadout
component (``readout``) for solution-space frontier readouts.  State,
grounding, recovery, and evidence semantics are unchanged.

V7 (additive): adds the three-operation facade ``NLSSModel`` and the first-class
evidence log.  No existing API is removed.

V7 Core (additive): adds the logic-governed neuro-symbolic representation layer
— ``RepresentationSpecification`` (P_t), ``LogicGate`` (L_tau) + ``FidelityGate``
(F_t) forming the dual admission gate, ``RepresentationRuntime`` (C_t),
``RepresentationParadigm`` (the V1/V6/V7-minus-logic/Full-V7 ablation arms), the
bidirectional ``LanguageCorrespondence`` (Gamma_t), and the evidence-driven
``RepresentationRevisioner`` (four revision triggers).  No existing API removed.
"""

from .model import (
    EvidenceLog,
    EvidenceRecord,
    LanguageCorrespondence,
    NLSSModel,
)
from .agent import NLSSAgentLoop, default_evaluator, default_implementer
from .reground import (
    DEFAULT_MISSING_DATA_POLICY,
    MISSING_DATA_POLICY,
    UNRECOVERABLE,
    ReGroundingPass,
    ReGroundingReport,
    UnknownAttribute,
    apply_missing_data_policy,
)
from .acquisition import (
    COV,
    BND,
    SOL,
    OPEN,
    CHANNELS,
    ExplorationController,
    ExplorationGear,
    GearProfile,
)
from .readout import (
    CoverageFrontierReadout,
    CoveredRegion,
    FrontierSemantics,
    FrontierTarget,
    ReadoutPacket,
    SpaceReadout,
    descriptor_distance,
    descriptor_tuple,
    render_readout,
)
from .representation import (
    FidelityGate,
    FidelityCertificate,
    LogicGate,
    LogicReport,
    LogicViolation,
    RelationDecl,
    RepresentationInducer,
    RepresentationParadigm,
    RepresentationRuntime,
    RepresentationSpecification,
    RuntimeRegistration,
    TransformDecl,
    TypedVariable,
    default_representation_spec,
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
)
from ..recovery.base import RecoveryInput

__all__ = [
    # V7 facade
    "NLSSModel",
    "EvidenceLog",
    "EvidenceRecord",
    "LanguageCorrespondence",
    "RecoveryInput",
    # V6.1 readout
    "SpaceReadout",
    "CoverageFrontierReadout",
    "CoveredRegion",
    "FrontierTarget",
    "ReadoutPacket",
    "FrontierSemantics",
    "descriptor_distance",
    "descriptor_tuple",
    "render_readout",
    # V7 Core representation layer
    "RepresentationParadigm",
    "RepresentationSpecification",
    "TypedVariable",
    "RelationDecl",
    "TransformDecl",
    "LogicGate",
    "LogicReport",
    "LogicViolation",
    "FidelityGate",
    "FidelityCertificate",
    "RepresentationRuntime",
    "RuntimeRegistration",
    "RepresentationInducer",
    "default_representation_spec",
    "RepresentationRevisioner",
    "RevisionTrigger",
    "RevisionSignal",
    "RevisionReport",
    # exploration dial
    "ExplorationGear",
    "GearProfile",
    "ExplorationController",
    "SOL",
    "BND",
    "COV",
    "OPEN",
    "CHANNELS",
    # Algorithm-1 loop + re-grounding
    "NLSSAgentLoop",
    "default_implementer",
    "default_evaluator",
    "ReGroundingPass",
    "ReGroundingReport",
    "UnknownAttribute",
    "MISSING_DATA_POLICY",
    "DEFAULT_MISSING_DATA_POLICY",
    "UNRECOVERABLE",
    "apply_missing_data_policy",
    # state / types
    "SolutionSpaceState",
    "GroundedObject",
    "LanguageHypothesis",
    "Observation",
]
