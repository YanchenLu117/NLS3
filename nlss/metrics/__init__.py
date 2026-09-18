"""NLSS metrics.

Exposes the solution-space recovery metrics from :mod:`nlss.metrics.recovery`
(including AURC, the area-under-recovery-vs-budget curve) plus calibration and
relational metrics, and the construction-track unified primary Recovery-AURC
(area under risk-vs-coverage, EXPERIMENT_PLAN §8.1) from :mod:`recovery_aurc`.
"""

from .recovery import (
    aurc,
    local_smoothness,
    normalized_dirichlet_energy,
    rmse,
    spearman_rank,
    top_k_auprc,
    top_k_recall,
    top_region_enrichment,
)
from .recovery_aurc import from_state, recovery_aurc, recovery_curve

__all__ = [
    "aurc",
    "local_smoothness",
    "normalized_dirichlet_energy",
    "rmse",
    "spearman_rank",
    "top_k_auprc",
    "top_k_recall",
    "top_region_enrichment",
    "recovery_curve",
    "recovery_aurc",
    "from_state",
]
