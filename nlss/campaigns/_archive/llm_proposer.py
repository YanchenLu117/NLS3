"""LLM causal-control proposer arms for the BH campaign (§9.8 B1/B2).

B1-History / B2-Scalar are the **same-backbone causal controls**: the same LLM
(DeepseekV4Flash) sees exactly the same raw measured history as the NLSS arm,
rendered through the repo's ``state_strategies`` (History / ScalarField), and
proposes the next batch of BH candidates ``a|l|b|d``.  The only difference from
the NLSS arm is the absence of the recovered solution-space readout — isolating
"is it the recovered state, or just a longer/better prompt".

Design choices for cost + fairness:
  * ONE generation per round returns the whole batch (JSON array of up to
    ``n`` candidates), then parsed + validated against the oracle pool.
  * Parsed candidates outside the measured pool or already queried are
    rejected and (only to keep the oracle budget exactly equal) replaced by
    random unqueried candidates — never a silent substitution of *another*
    proposed candidate; the arm's own proposals that are valid go through.
  * Token ledger records every generation through ResourceLedger.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Sequence

from nlss.core.state import SolutionSpaceState
from nlss.state_strategies.history import HistoryState
from nlss.state_strategies.scalar import ScalarFieldState


class LLMProposer:
    """Drives DeepseekV4Flash to propose BH candidate batches from a rendered state."""

    _COMPONENT_KEYS = ("aryl_halide", "ligand", "base", "additive")

    def __init__(
        self,
        strategy: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 8192,  # DeepseekV4Flash requires [8192, 32768]
        vocab: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        from nlss.llm.deepseek import DeepseekV4Flash

        if strategy not in ("history", "scalar"):
            raise ValueError(strategy)
        self.strategy_name = strategy
        self._rendered = HistoryState() if strategy == "history" else ScalarFieldState()
        self._vocab = vocab or {}
        self._provider = DeepseekV4Flash(
            api_key=api_key or os.environ.get("NLSS_LLM_API_KEY", ""),
            base_url=base_url or os.environ.get("NLSS_LLM_BASE_URL"),
            model=model,
            max_output_tokens=max_output_tokens,
        )
        self._temperature = temperature

    # -- rendering ----------------------------------------------------------

    def render(
        self,
        state: SolutionSpaceState,
        adapter,
        token_budget: int = 1200,
    ) -> str:
        import asyncio

        return asyncio.run(self._rendered.render_state(state, adapter, self._provider, token_budget))

    # -- proposal -----------------------------------------------------------

    def _system_prompt(self) -> str:
        keys = self._COMPONENT_KEYS
        vocab_desc = "; ".join(
            f"{k}: {sorted(v)[:8]}{'...' if len(v) > 8 else ''}" for k, v in self._vocab.items() if v
        )
        return (
            "You are an experimental chemist running Buchwald-Hartwig C-N cross-coupling "
            "optimization. Propose NEW reaction conditions to maximize yield.\n"
            "A candidate is exactly four pipe-separated fields: "
            "aryl_halide|ligand|base|additive (additive may be empty).\n"
            "Component vocabularies (extract):\n" + vocab_desc + "\n"
            "Output ONLY a JSON array of candidate strings, e.g. [\"A|B|C|D\", ...]. "
            "Each candidate must use only the given vocab, be structurally varied, "
            "and NOT duplicate candidates already tried."
        )

    def propose(self, context: str, n: int) -> tuple[list[str], dict]:
        """Ask the LLM for n candidates; returns (parsed_candidates, usage)."""
        user = (
            context
            + f"\n\nPropose exactly {n} new, distinct candidate string(s) now. "
            + "Respond with valid JSON array only."
        )
        gen = self._provider.generate(self._system_prompt(), user, temperature=self._temperature)
        parsed = self._parse_json_array(gen.text)
        return parsed, {
            "input_tokens": gen.input_tokens,
            "output_tokens": gen.output_tokens,
            "total_tokens": gen.total_tokens,
            "raw": gen.text[:200],
        }

    @staticmethod
    def _parse_json_array(text: str) -> list[str]:
        stripped = text.strip()
        # strip fenced code block if present
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1]).strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                return [str(s).strip() for s in data if isinstance(s, (str, int, float))]
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return [str(s).strip() for s in v]
            return []
        except json.JSONDecodeError:
            # best-effort: extract quoted strings
            return re.findall(r'"([^"]+)"', stripped)

    # -- validation ---------------------------------------------------------

    def validate(
        self,
        proposed: Sequence[str],
        pool: set,
        queried: set,
        n: int,
        rng,
    ) -> list[Any]:
        """Keep valid unqueried candidates; pad with random so budget is exact."""
        parsed_cands = []
        for s in proposed:
            parts = [p.strip() for p in s.split("|")]
            if len(parts) < 3:
                continue
            cand = tuple(parts[:4]) if len(parts) >= 4 else (parts[0], parts[1], parts[2], "")
            if cand in pool and cand not in queried:
                parsed_cands.append(cand)
        # dedup preserving order, then pad to n with random unqueried
        seen = set(queried)
        keep = [c for c in parsed_cands if c not in seen and not (seen.add(c) or False)]
        pad_pool = [c for c in pool if c not in seen]
        rng.shuffle(pad_pool)
        keep.extend(pad_pool[: n - len(keep)])
        return keep[:n]
