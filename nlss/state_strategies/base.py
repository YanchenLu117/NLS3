"""StateStrategy — the baseline-fairness seam.

History / Reflection / Population / Scalar / ShuffledRelational / FullNLSS all
render the same SolutionSpaceState into a token-budgeted prompt string.  This is
the single place that makes the "History vs ... vs NLSS" comparison come from
one implementation instead of four benchmark-local copies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core.interfaces import TaskAdapter
    from ..core.state import SolutionSpaceState
    from ..llm.base import LLMProvider


class StateStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def render_state(
        self,
        state: "SolutionSpaceState",
        adapter: "TaskAdapter",
        llm: "LLMProvider",
        token_budget: int,
    ) -> str: ...
