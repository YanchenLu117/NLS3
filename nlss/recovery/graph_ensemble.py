"""Neural graph ensemble backend (B5) — GraphSAGE x 5 ensembles.

Phase 1 stub: implemented in the Phase 2 Static Recovery Bake-off (requires
torch / torch-geometric; kept lazy so the core stays importable).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.errors import RecoveryNotFitted
from .base import RecoveryBackend, RecoveryInput


class GraphEnsembleBackend(RecoveryBackend):
    name = "graph_ensemble"

    def __init__(self, *, n_ensembles: int = 5) -> None:
        self.n_ensembles = n_ensembles

    def fit(self, recovery_input: RecoveryInput) -> None:
        raise NotImplementedError(
            "graph ensemble is implemented in Phase 2 bake-off"
        )

    def posterior(self):
        raise RecoveryNotFitted("graph ensemble is not fitted (Phase 2)")

    def diagnostics(self) -> Mapping[str, Any]:
        return {"backend": self.name, "status": "stub_phase2"}
