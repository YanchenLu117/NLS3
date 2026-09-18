"""Scalar field state strategy — same posterior as Full NLSS, but exposes only
marginal candidate information (no relational evidence)."""

from __future__ import annotations

from ..core.errors import RecoveryNotFitted
from .base import StateStrategy


class ScalarFieldState(StateStrategy):
    name = "scalar_field"

    async def render_state(self, state, adapter, llm, token_budget: int) -> str:
        if state.posterior is None:
            raise RecoveryNotFitted("scalar state requires a fitted posterior")
        lines = ["## Marginal candidate scores (mean, uncertainty)"]
        items = []
        for oid, obj in state.objects.items():
            m = state.posterior.marginal(oid)
            items.append((m.mean, oid, obj, m.std))
        items.sort(key=lambda t: t[0], reverse=True)
        for mean, oid, obj, std in items:
            marker = "*" if state.is_observed(oid) else " "
            lines.append(f"- [{marker}] {adapter.render_object(obj)}: {mean:.4g} +- {std:.4g}")
        return "\n".join(lines)[: token_budget * 4]
