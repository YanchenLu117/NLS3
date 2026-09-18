"""Solution-Space Reconstruction — render (Agent D, read-only consumer).

Generic 2D projection + four colourings:
  empirical Q (posterior mean), region, round-discovered, truth-distance.
For MADE we draw a ternary simplex; the generic scatter works for any 2D map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import RecoveredSolutionNode


def _hex_region(r):
    # deterministic colour per region id
    h = 0
    for ch in str(r):
        h = (h * 31 + ord(ch)) & 0xFFFFFF
    return "#%06x" % (h % 0xCCCCCC + 0x111111)


def render_map(nodes: Sequence[RecoveredSolutionNode],
               regions=None,
               boundaries=None,
               frontiers=None,
               *,
               colour_by: str = "region",
               truth_distance: Mapping[str, float] | None = None,
               out: str | Path | None = None,
               title: str | None = None,
               show_region_labels: bool = True):
    """Draw a 2D reconstruction map.

    colour_by: 'q' (empirical Q), 'region', 'round', 'truth'.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]

    if colour_by == "q":
        vals = [n.mean_utility if n.mean_utility is not None else 0.0 for n in nodes]
        sc = ax.scatter(xs, ys, c=vals, cmap="viridis", s=30, alpha=0.8, edgecolors="k", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label="empirical Q")
    elif colour_by == "round":
        vals = [n.round_index if n.round_index is not None else -1 for n in nodes]
        sc = ax.scatter(xs, ys, c=vals, cmap="plasma", s=30, alpha=0.8, edgecolors="k", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label="round discovered")
    elif colour_by == "truth":
        if truth_distance is None:
            truth_distance = {}
        vals = [truth_distance.get(n.object_id, np.nan) for n in nodes]
        sc = ax.scatter(xs, ys, c=vals, cmap="coolwarm", s=30, alpha=0.8, edgecolors="k", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label="truth distance")
    else:  # region
        colours = [_hex_region(n.region_id) if n.region_id else "#cccccc" for n in nodes]
        ax.scatter(xs, ys, c=colours, s=30, alpha=0.8, edgecolors="k", linewidths=0.3)

    # regions as hull-ish circles
    if regions and show_region_labels:
        for r in regions:
            ax.scatter([r.center_x], [r.center_y], marker="o", s=80, facecolors="none",
                       edgecolors=_hex_region(r.region_id), linewidths=1.2)
            ax.annotate(r.region_id, (r.center_x, r.center_y), fontsize=8)
    # boundaries
    if boundaries:
        for b in boundaries:
            for nid in getattr(b, "node_ids", ()):
                # find node
                pass  # visual marker omitted for simplicity (boundary nodes coloured by node list)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title or "Recovered solution-space map")
    ax.set_aspect("equal", adjustable="box")
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out)
    return fig
