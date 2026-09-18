"""Reflection state strategy — history + a reflection prompt."""

from __future__ import annotations

from .base import StateStrategy


class ReflectionState(StateStrategy):
    name = "reflection"

    async def render_state(self, state, adapter, llm, token_budget: int) -> str:
        obs = sorted(state.observations.values(), key=lambda o: o.round_index)
        lines = ["## Observed history"]
        for o in obs:
            obj = state.objects.get(o.object_id)
            rep = adapter.render_object(obj) if obj else o.object_id
            lines.append(f"- r{o.round_index}: {rep} -> {o.value:.4g}")
        lines.append("## Reflect on the observed history and propose the next hypothesis.")
        return "\n".join(lines)[: token_budget * 4]
