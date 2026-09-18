"""V8 P0-b — statistical decision layer (Plan §3.1/§12, Detail §14)."""

from .decision import (
    DECISION_EQUIVALENT,
    DECISION_HARMFUL,
    DECISION_IMPROVES,
    DECISION_INCONCLUSIVE,
    Decision,
    PairedSummary,
    bootstrap_ci,
    cluster_bootstrap_ci,
    decide,
    hierarchical_bootstrap_ci,
    holm_adjust,
    paired_median_ci,
    paired_summary,
    smd_serializable,
    tost_equivalent,
)

__all__ = [
    "DECISION_EQUIVALENT",
    "DECISION_HARMFUL",
    "DECISION_IMPROVES",
    "DECISION_INCONCLUSIVE",
    "Decision",
    "PairedSummary",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "decide",
    "hierarchical_bootstrap_ci",
    "holm_adjust",
    "paired_median_ci",
    "paired_summary",
    "smd_serializable",
    "tost_equivalent",
]
