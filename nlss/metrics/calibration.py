"""Probabilistic calibration metrics (Phase 2 §2.6).

Two notions of calibration for a recovery posterior:

* **interval coverage** — the empirical frequency with which the true value lands
  inside the marginal ``mean ± z·std`` interval (should track ``Φ(z) - Φ(-z)``).
* **expected calibration error (ECE)** — binned gap between the posterior's
  solution probability and the empirical solution frequency.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def interval_coverage(
    true_values: Mapping[str, float],
    means: Mapping[str, float],
    stds: Mapping[str, float],
    *,
    z: float = 1.96,
) -> float:
    """Fraction of nodes whose true value lies within ``mean ± z·std``."""
    ids = [k for k in true_values if k in means and k in stds]
    if not ids:
        return float("nan")
    hit = 0
    for nid in ids:
        t = true_values[nid]
        m = means[nid]
        s = stds[nid]
        if not (np.isfinite(t) and np.isfinite(m) and np.isfinite(s)):
            continue
        hit += int(abs(t - m) <= z * s)
    return float(hit / len(ids))


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    """Binned ECE = Σ (|B_m|/N) · |acc(B_m) - conf(B_m)|.

    ``labels`` are binary (solution or not); ``probabilities`` are the posterior's
    predicted solution probability for the same items.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            in_bin = (p >= lo) & (p <= 1.0)
        else:
            in_bin = (p >= lo) & (p < hi)
        if not in_bin.any():
            continue
        acc = float(y[in_bin].mean())
        conf = float(p[in_bin].mean())
        ece += (in_bin.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def sharpness(stds: Mapping[str, float]) -> float:
    """Mean marginal posterior std — lower is sharper (reported with coverage)."""
    vals = [s for s in stds.values() if np.isfinite(s)]
    return float(np.mean(vals)) if vals else float("nan")
