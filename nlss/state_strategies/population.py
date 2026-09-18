"""Population state strategy — render the current candidate population."""

from __future__ import annotations

from .base import StateStrategy


class PopulationState(StateStrategy):
    name = "population"

    async def render_state(self, state, adapter, llm, token_budget: int) -> str:
        lines = ["## Current candidate population"]
        for oid, obj in state.objects.items():
            marker = "*" if state.is_observed(oid) else " "
            lines.append(f"- [{marker}] {adapter.render_object(obj)}")
        return "\n".join(lines)[: token_budget * 4]
