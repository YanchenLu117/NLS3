"""Full NLSS state strategy — renders the shared EvidencePacket."""

from __future__ import annotations

from ..core.evidence import EvidenceBuilder
from .base import StateStrategy


class FullNLSSState(StateStrategy):
    name = "nlss_full"

    def __init__(self, evidence_builder: EvidenceBuilder | None = None) -> None:
        self.evidence_builder = evidence_builder or EvidenceBuilder()

    async def render_state(self, state, adapter, llm, token_budget: int) -> str:
        packet = self.evidence_builder.build(state, adapter)
        lines = [
            "## Anchors (high empirical utility)",
        ]
        for a in packet.anchors:
            lines.append(f"- {a.representation} (mean {a.mean_utility:.4g})")
        lines.append("## Alternatives (distinct regions)")
        for a in packet.alternatives:
            lines.append(f"- {a.representation} (mean {a.mean_utility:.4g})")
        lines.append("## Boundaries (high uncertainty)")
        for b in packet.boundaries:
            lines.append(f"- {b.representation} (uncertainty {b.uncertainty:.4g})")
        lines.append("## Revisions (source -> target: action)")
        for r in packet.revisions:
            lines.append(
                f"- {r.source_repr} -> {r.target_repr} [{r.action_type}]: "
                f"delta {r.mean_delta:.4g} +- {r.std_delta:.4g} "
                f"(improvement {r.improvement_probability:.3g})"
            )
        return "\n".join(lines)[: token_budget * 4]
