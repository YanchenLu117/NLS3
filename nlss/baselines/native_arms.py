"""V7 Native AI-scientist / host arms (EXPERIMENT_PLAN §5.2 / §6.2).

The plan treats the official AI-scientist HOSTS as baselines in the native
regime: SciExplorer and MADE (four arms) and PiEvo (three arms).  These are the
"AI-scientist baselines" — distinct from §5.4 related-work-only published papers
(DeepScientist/CodeScientist/SR-Scientist/EvoDiverse/BALLET/LGBO/RankFlow) which
are excluded from the quantitative matrix.

This module structures every host's arm enum + injection semantics in ONE place
so execution-team runners (one per host) are uniform and auditable.  Live
execution requires the official repo (see DATA_MANIFEST.md); these are the
control surfaces, not the runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class Host(str, Enum):
    SCIEXPLORER = "sciexplorer"   # four arms (Native/Summary/NLSS-State/Full)
    MADE = "made"                 # four arms
    PIEVO = "pievo"               # three arms (Native/Summary/NLSS-plug-in)


@dataclass(frozen=True, slots=True)
class NativeArmSpec:
    host: Host
    arms: tuple[str, ...]
    official_repo: str
    note: str
    injection: Mapping[str, str]


_NATIVE_ARMS: Mapping[Host, NativeArmSpec] = {
    Host.SCIEXPLORER: NativeArmSpec(
        Host.SCIEXPLORER, ("native", "summary", "nlss_state", "full"),
        "github.com/MaxNaeg/SciExplorer",
        "sidecar over run_exploration public interfaces; smoke_gate_ok before official numbers (§3.1)",
        {"native": "official loop unchanged", "summary": "token-matched prose summary context only",
         "nlss_state": "read-only NLSS state in fields/round message", "full": "NLSS channel + target region injected per round"}),
    Host.MADE: NativeArmSpec(
        Host.MADE, ("native", "summary", "nlss_state", "full"),
        "github.com/diffractivelabs/MADE",
        "official Planner/Generator/Filter/Scorer extension interfaces; State-Summary=rep utility, Full-State=decision utility (§3.2)",
        {"native": "official pipeline unchanged", "summary": "token-matched prose summary context only",
         "nlss_state": "state['nlss_readout'] added for official planner/scorer", "full": "NLSSPlanner + NLSSFieldScorer replace official proposer/selector"}),
    Host.PIEVO: NativeArmSpec(
        Host.PIEVO, ("native", "summary", "nlss_plugin"),
        "github.com/eurekaw/pievo",
        "multi-agent host; transfer panel only, NOT in single-agent headline ranking (§4.7)",
        {"native": "official multi-agent loop unchanged", "summary": "token-matched evidence summary only",
         "nlss_plugin": "NLSS state as control surface for the discovery loop (C5)"}),
}


def host_arms(host: Host) -> tuple[str, ...]:
    return _NATIVE_ARMS[host].arms


def arm_injection(host: Host, arm: str) -> str:
    spec = _NATIVE_ARMS[host]
    return spec.injection.get(arm, f"unspecified ({arm})")


def native_hosts() -> tuple[Host, ...]:
    return tuple(Host)


def spec(host: Host) -> NativeArmSpec:
    return _NATIVE_ARMS[host]


def native_baselines_index() -> dict[str, dict[str, Any]]:
    """Julian-less human/runner index of all native AI-scientist arms (§5.2)."""
    out: dict[str, dict[str, Any]] = {}
    for host in native_hosts():
        s = _NATIVE_ARMS[host]
        out[host.value] = {
            "arms": list(s.arms), "official_repo": s.official_repo, "note": s.note,
            "injection": dict(s.injection),
        }
    return out


# ---- Tier B: recent quantitative AI-scientist systems (EXPERIMENT_... §3.2) --

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AIScientistSpec:
    key: str            # S1..S4
    reported_name: str
    source: str
    frozen_placement: str
    tool_boundary: str  # how it touches the shared campaign API / adapter


AI_SCIENTIST_SYSTEMS: Mapping[str, AIScientistSpec] = {
    "S1": AIScientistSpec("S1", "SciExplorer", "github.com/MaxNaeg/SciExplorer", "SciExplorer-Physics",
                          "four-arm Native/Summary/NLSS-State/Full via run_exploration public interface; smoke_gate_ok"),
    "S2": AIScientistSpec("S2", "AI Scientist-v2-Discovery", "github.com/SakanaAI/AI-Scientist-v2", "GB1 + MADE",
                          "shared campaign API GB1/MADE; preserve native tree-search; only experiment boundary adapted"),
    "S3": AIScientistSpec("S3", "AI-Researcher-Discovery", "github.com/HKUDS/AI-Researcher", "GB1 + MADE",
                          "shared campaign API GB1/MADE; preserve multi-stage components; report component/cost"),
    "S4": AIScientistSpec("S4", "SR-Scientist", "github.com/GAIR-NLP/SR-Scientist", "applicable SciExplorer-Physics tasks",
                          "executable-equation interface only; applicability recorded pre-outcome, else N/A"),
}


def ai_scientist_systems() -> dict[str, dict[str, str]]:
    return {k: {"reported_name": v.reported_name, "source": v.source,
                "frozen_placement": v.frozen_placement, "tool_boundary": v.tool_boundary}
            for k, v in AI_SCIENTIST_SYSTEMS.items()}


def ai_scientist_keys() -> tuple[str, ...]:
    return tuple(sorted(AI_SCIENTIST_SYSTEMS))
