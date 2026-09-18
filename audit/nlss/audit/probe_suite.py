"""probe_suite.py — frozen + fresh probe suites (Main Theorem II mechanism).

SUITES (prereg-frozen per task family) are versioned probe batteries with
hashes. Each suite:
  - covers every audit dimension with concrete probes (payload + rubric id);
  - has suite_id = sha256 over canonical probe list;
  - supports FRESH generation: new probes drawn from the held-out probe bank,
    never shown to the candidate's proposer session, never reused across
    certified activations (fresh/rotating held-out probes → conditional
    validity for activation).

Anti-gaming rules (frozen):
  PS1 probe bank is evaluator-only; proposer prompts never include probe text;
  PS2 a suite certifies exactly ONE activation (one-shot);
  PS3 reuse arms (S4) consume only suites already burned on OTHER candidates;
  PS4 rotation log is append-only: suite_id, candidate_id, burned_at.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

ROTATION_LOG = Path("runs/probe_rotation.jsonl")


@dataclass
class Probe:
    probe_id: str
    dimension: str          # one of SCORE_DIMS
    kind: str               # "judge" | "exec"
    payload: dict           # probe content (for judge: rubric_id + inputs; exec: args)
    bank_ref: str = ""      # provenance into held-out bank


@dataclass
class Suite:
    suite_id: str
    probes: list = field(default_factory=list)
    fresh_for: str = ""     # candidate_id it was freshly drawn for
    burned: bool = False


def suite_id_for(probes: list) -> str:
    blob = json.dumps([p.payload for p in probes], sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return "s_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def build_suite(bank: list, dims: list, rng: random.Random,
                per_dim: int = 2) -> Suite:
    """Draw a fresh suite from the held-out probe bank: per_dim probes per dim."""
    probes: list[Probe] = []
    by_dim: dict = {}
    for p in bank:
        by_dim.setdefault(p["dimension"], []).append(p)
    for dim in dims:
        pool = by_dim.get(dim, [])
        rng.shuffle(pool)
        for src in pool[:per_dim]:
            probes.append(Probe(probe_id=src["probe_id"], dimension=dim,
                                kind=src.get("kind", "judge"),
                                payload=src.get("payload", {}),
                                bank_ref=src.get("bank_ref", "")))
    return Suite(suite_id=suite_id_for(probes), probes=probes)


def burn(suite: Suite, candidate_id: str, log_path: Path = ROTATION_LOG) -> None:
    """One-shot certification: burn the suite and append to the rotation log."""
    suite.burned = True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as g:
        g.write(json.dumps({"suite_id": suite.suite_id,
                            "candidate_id": candidate_id,
                            "burned_utc": _utc()}, ensure_ascii=False) + "\n")


def is_fresh(suite_id: str, log_path: Path = ROTATION_LOG) -> bool:
    if not log_path.exists():
        return True
    return suite_id not in {json.loads(l)["suite_id"]
                            for l in open(log_path) if l.strip()}


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
