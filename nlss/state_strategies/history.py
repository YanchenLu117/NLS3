"""History state strategy — trajectory/history baseline."""

from __future__ import annotations

from .base import StateStrategy


class HistoryState(StateStrategy):
    name = "history"

    async def render_state(self, state, adapter, llm, token_budget: int) -> str:
        lines = ["## Observed history (round, object, utility)"]
        obs = sorted(state.observations.values(), key=lambda o: o.round_index)
        for o in obs:
            obj = state.objects.get(o.object_id)
            rep = adapter.render_object(obj) if obj else o.object_id
            lines.append(f"- r{o.round_index}: {rep} -> {o.value:.4g}")
        return "\n".join(lines)[: token_budget * 4]
