"""Shuffled relational state strategy — Full NLSS packet with permuted relation
targets.  Preserves packet size/token count/numeric distribution while breaking
relation correctness; this isolates the value of relational information."""

from __future__ import annotations

import random

from ..core.evidence import EvidenceBuilder
from .base import StateStrategy


class ShuffledRelationalState(StateStrategy):
    name = "shuffled_relational"

    def __init__(
        self,
        evidence_builder: EvidenceBuilder | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.seed = seed

    async def render_state(self, state, adapter, llm, token_budget: int) -> str:
        packet = self.evidence_builder.build(state, adapter)
        # Permute revision target/source among the revision list to break only
        # relation correctness (size and numeric distribution preserved).
        rng = random.Random(self.seed + state.round_index)
        revs = list(packet.revisions)
        if len(revs) > 1:
            source_ids = [r.source_id for r in revs]
            rng.shuffle(source_ids)
            revs = [
                r.__class__(
                    source_id=sid,
                    target_id=r.target_id,
                    source_repr=r.source_repr,
                    target_repr=r.target_repr,
                    action_type=r.action_type,
                    action_description=r.action_description,
                    mean_delta=r.mean_delta,
                    std_delta=r.std_delta,
                    improvement_probability=r.improvement_probability,
                    confidence_label=r.confidence_label,
                    metadata=r.metadata,
                )
                for sid, r in zip(source_ids, revs)
            ]

        lines = ["## Revisions (source -> target: action) [relation-shuffled]"]
        for r in revs:
            lines.append(
                f"- {r.source_repr} -> {r.target_repr} [{r.action_type}]: "
                f"delta {r.mean_delta:.4g} (improvement {r.improvement_probability:.3g})"
            )
        return "\n".join(lines)[: token_budget * 4]
