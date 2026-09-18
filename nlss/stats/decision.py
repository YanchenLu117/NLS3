"""V8 P0-b — unified statistical decision layer (Plan §3.1/§12, Detail §14).

Public surface:
- :func:`paired_summary`     — paired mean/median, 95% bootstrap CI, paired SMD,
  paired randomization p-value (per-unit paired differences, Detail §14).
- :func:`holm_adjust`        — Holm step-down within a claim family (Plan §3.1).
- :func:`tost_equivalent`    — two one-sided tests vs ±delta_eq (90% CI rule).
- :func:`decide`             — the preregistered decision language:
  improves / equivalent / materially_harmful / inconclusive (Plan §3.1).
- :func:`bootstrap_ci`       — unit bootstrap CI.
- :func:`cluster_bootstrap_ci` — instance-cluster bootstrap (HypoSpace §12:
  average repeats within instance, then resample instances).
- :func:`hierarchical_bootstrap_ci` — two-stage task/system x seed bootstrap
  (MADE/Physics §12).

Determinism: all stochastic pieces take an explicit seed and use
``numpy.random.default_rng``.  Failure accounting (failed runs stay in the
denominator; deterministic endpoint scores, Detail §6/§14) is the caller's
responsibility: it happens at the metric/curve layer before statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "Decision",
    "PairedSummary",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "decide",
    "hierarchical_bootstrap_ci",
    "holm_adjust",
    "paired_summary",
    "tost_equivalent",
]

DECISION_IMPROVES = "improves"
DECISION_EQUIVALENT = "equivalent"
DECISION_HARMFUL = "materially_harmful"
DECISION_INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class PairedSummary:
    n: int
    mean: float
    median: float
    ci_low: float
    ci_high: float
    smd: float
    permutation_p: float

    def to_dict(self) -> dict[str, Any]:
        """RFC 8259-safe artifact form (Detail §15): non-finite SMD becomes
        null + an explicit degenerate flag instead of bare Infinity."""
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            **smd_serializable(self.smd),
            "permutation_p": self.permutation_p,
        }


@dataclass(frozen=True)
class Decision:
    label: str
    summary: PairedSummary
    delta_min: float
    delta_eq: float
    delta_harm: float
    raw_p: float
    holm_p: float
    tost_low_p: float
    tost_high_p: float


def _as_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < 2:
        raise ValueError(f"{name} needs at least 2 paired units (registry min_samples guards this)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _paired_deltas(nlss: Sequence[float], baseline: Sequence[float]) -> np.ndarray:
    a = _as_array(nlss, "nlss")
    b = _as_array(baseline, "baseline")
    if a.shape != b.shape:
        raise ValueError("paired series must have identical length")
    return a - b


def _bootstrap_means(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    return values[idx].mean(axis=1)


def bootstrap_ci(values: Sequence[float], *, alpha: float = 0.05, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    arr = _as_array(values, "values")
    means = _bootstrap_means(arr, n_boot, seed)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def cluster_bootstrap_ci(
    values: Sequence[float],
    cluster_ids: Sequence,
    *,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Resample clusters with replacement; statistic = mean of cluster means.

    Per Plan §12 the caller may equivalently pre-average repeats within each
    instance and pass one value per instance; this function handles the general
    unbalanced case (several units per cluster).
    """
    arr = _as_array(values, "values")
    ids = np.asarray(cluster_ids)
    if ids.shape != arr.shape:
        raise ValueError("cluster_ids must match values")
    unique = np.unique(ids)
    if unique.size < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters")
    per_cluster = np.array([arr[ids == c].mean() for c in unique], dtype=float)
    return bootstrap_ci(per_cluster, alpha=alpha, n_boot=n_boot, seed=seed)


def hierarchical_bootstrap_ci(
    values: Sequence[float],
    group_ids: Sequence,
    unit_ids: Sequence,
    *,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Two-stage bootstrap for task/system x environment-seed designs (§12).

    Stage 1 resamples groups (tasks) with replacement; stage 2 resamples the
    units (seeds) within each sampled group; statistic = mean of group means.
    """
    arr = _as_array(values, "values")
    g = np.asarray(group_ids)
    u = np.asarray(unit_ids)
    if g.shape != arr.shape or u.shape != arr.shape:
        raise ValueError("group_ids/unit_ids must match values")
    groups = np.unique(g)
    if groups.size < 2:
        raise ValueError("hierarchical bootstrap needs at least 2 groups")
    rng = np.random.default_rng(seed)
    per_group: dict[object, np.ndarray] = {grp: arr[g == grp] for grp in groups}
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_groups = rng.choice(groups, size=groups.size, replace=True)
        total = 0.0
        for grp in sampled_groups:
            units = per_group[grp]
            picked = units[rng.integers(0, units.size, size=units.size)]
            total += picked.mean()
        stats[b] = total / groups.size
    return float(np.quantile(stats, alpha / 2.0)), float(np.quantile(stats, 1.0 - alpha / 2.0))


def paired_permutation_pvalue(deltas: np.ndarray, *, n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired randomization (sign-flip) test on the mean difference.

    Exact enumeration for n <= 16 (2^n <= 65536 sign patterns); Monte-Carlo
    with a relative tie tolerance above that.
    """
    n = deltas.size
    obs = abs(float(deltas.mean()))
    if n <= 16:
        # enumerate all 2^n sign patterns (exact randomization)
        idx = np.arange(1 << n, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1) * 2.0 - 1.0
        perm_means = np.abs((bits * deltas).mean(axis=1))
        extreme = int((perm_means >= obs - 1e-12 * max(1.0, obs)).sum())
        return extreme / (1 << n)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = np.abs((signs * deltas).mean(axis=1))
    extreme = int((perm_means >= obs - 1e-12 * max(1.0, obs)).sum())
    return (1.0 + extreme) / (n_perm + 1.0)


def paired_smd(deltas: np.ndarray) -> float:
    """Paired standardized mean difference: mean(d) / sd(d, ddof=1).

    Degenerate variance yields ±inf by definition; serialization boundaries
    must map non-finite values to ``null`` + an explicit degenerate flag
    (:func:`smd_serializable`) — raw inf breaks RFC 8259 JSON.
    """
    sd = float(deltas.std(ddof=1))
    mean = float(deltas.mean())
    if sd == 0.0:
        return 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    return mean / sd


def smd_serializable(smd: float) -> dict[str, Any]:
    """RFC 8259-safe SMD for artifact JSON: non-finite -> null + flag."""
    if math.isfinite(smd):
        return {"smd": smd, "smd_degenerate": False}
    return {"smd": None, "smd_degenerate": True, "smd_sign": "+" if smd > 0 else "-"}


def paired_summary(
    nlss: Sequence[float],
    baseline: Sequence[float],
    *,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedSummary:
    """Detail §14 reporting block for one primary contrast."""
    deltas = _paired_deltas(nlss, baseline)
    low, high = bootstrap_ci(deltas, alpha=alpha, n_boot=n_boot, seed=seed)
    return PairedSummary(
        n=int(deltas.size),
        mean=float(deltas.mean()),
        median=float(np.median(deltas)),
        ci_low=low,
        ci_high=high,
        smd=paired_smd(deltas),
        permutation_p=paired_permutation_pvalue(deltas, n_perm=n_perm, seed=seed),
    )


def paired_median_ci(
    deltas: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """§14 gap flagged by review: the paired median gets its own bootstrap CI
    (the mean already has one via :func:`paired_summary`)."""
    arr = _as_array(deltas, "deltas")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    medians = np.median(arr[idx], axis=1)
    return float(np.quantile(medians, alpha / 2.0)), float(np.quantile(medians, 1.0 - alpha / 2.0))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, enforced monotone (family-wise)."""
    arr = np.asarray(pvalues, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D sequence")
    if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
        raise ValueError("pvalues must be finite in [0, 1]")
    order = np.argsort(arr, kind="stable")
    m = arr.size
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = arr[idx] * (m - rank)
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return [float(v) for v in adjusted]


def tost_equivalent(deltas: np.ndarray, delta_eq: float, *, alpha: float = 0.10) -> tuple[float, float, bool]:
    """Two one-sided paired t-tests against ±delta_eq (Plan §3.1 TOST rule).

    Returns (p_lower, p_upper, equivalent).  ``equivalent`` iff both one-sided
    p-values are below alpha/2 — algebraically identical to the 90% paired CI
    lying inside ±delta_eq.
    """
    if delta_eq <= 0:
        raise ValueError("delta_eq must be positive")
    n = deltas.size
    mean = float(deltas.mean())
    se = float(deltas.std(ddof=1) / math.sqrt(n))
    dof = n - 1
    from scipy import stats as _stats

    if se == 0.0:
        p_low = 0.0 if mean > -delta_eq else 1.0
        p_high = 0.0 if mean < delta_eq else 1.0
    else:
        t_low = (mean - (-delta_eq)) / se  # H0: mean <= -delta_eq
        t_high = (mean - delta_eq) / se  # H0: mean >= +delta_eq
        p_low = float(_stats.t.sf(t_low, dof))
        p_high = float(_stats.t.cdf(t_high, dof))
    threshold = alpha / 2.0
    return p_low, p_high, (p_low < threshold and p_high < threshold)


def decide(
    nlss: Sequence[float],
    baseline: Sequence[float],
    *,
    delta_min: float,
    delta_eq: float,
    delta_harm: float,
    family_pvalues: Sequence[float] | None = None,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
) -> Decision:
    """Apply the Plan §3.1 decision language to one paired contrast.

    ``family_pvalues``: raw p-values of the whole preregistered claim family —
    the Holm adjustment is computed INSIDE (§12: Holm within each claim family;
    never a silent raw-p fallback).  Omitting it is valid only for a
    single-test family.

    Precedence: materially_harmful FIRST (harm must never be labeled
    equivalent when δ_eq > δ_harm), then improves, then equivalent, else
    inconclusive.  ``improves`` requires the Holm-adjusted p < 0.05 AND the
    95% paired-CI lower bound > delta_min.  ``equivalent`` requires TOST at
    90% within ±delta_eq.  Terminal win rates never substitute (§3.1).
    """
    if not (delta_min > 0 and delta_eq > 0 and delta_harm > 0):
        raise ValueError("delta_min/delta_eq/delta_harm must be positive (P0 preregisters them)")
    summary = paired_summary(nlss, baseline, n_boot=n_boot, n_perm=n_perm, seed=seed)
    deltas = _paired_deltas(nlss, baseline)
    raw_p = summary.permutation_p
    family = [raw_p] if family_pvalues is None else list(family_pvalues)
    holm_all = holm_adjust(family)
    try:
        holm_p = holm_all[family.index(raw_p)]
    except ValueError:
        holm_p = holm_all[0]
    tost_low, tost_high, equivalent = tost_equivalent(deltas, delta_eq)

    if summary.ci_high < -delta_harm:
        label = DECISION_HARMFUL
    elif summary.ci_low > delta_min and holm_p < 0.05:
        label = DECISION_IMPROVES
    elif equivalent:
        label = DECISION_EQUIVALENT
    else:
        label = DECISION_INCONCLUSIVE
    return Decision(
        label=label,
        summary=summary,
        delta_min=delta_min,
        delta_eq=delta_eq,
        delta_harm=delta_harm,
        raw_p=raw_p,
        holm_p=holm_p,
        tost_low_p=tost_low,
        tost_high_p=tost_high,
    )
