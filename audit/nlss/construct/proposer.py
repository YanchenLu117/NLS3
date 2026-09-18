"""proposer.py — candidate proposer (Construct stage, Methods §1).

Role separation (C3): the proposer LLM sees ONLY the task evidence and the χ
schema description. It is never shown M_ref, menu words, audit probes, or the
verifier's internals. CI-enforced: PROMPT_TEMPLATE + any task-render text pass
check_menu_words() at module import (fail-fast) and in the CI test.

Actor path: any OpenAI-compatible endpoint via NLSS_LLM_BASE_URL; per-arm model; decode_seed → sampling
seed + distinct decodes for novelty measurement. Output must be a single JSON
object parseable into χ (validated by schema.chi_schema.validate).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request

from ..schema.chi_schema import validate, check_menu_words, SCHEMA_VERSION
from ..schema.chi_schema import ValidationResult

ACTOR_URL = os.environ.get("NLSS_LLM_BASE_URL", "")
API_KEY = os.environ.get("NLSS_LLM_API_KEY", "")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# ------------------------------------------------------------------ #
# PROMPT TEMPLATE — menu-word-free by construction (CI gate below).
# Describes WHAT the candidate must declare (semantic content, executable
# organization, realization, interpretation rules, claims); never suggests
# any particular representational family.
# ------------------------------------------------------------------ #
PROMPT_TEMPLATE = """You are designing a computational scientific substrate for one task.

The substrate must let an explorer do three things with the task's evidence:
(1) express the scientifically meaningful hypotheses and how they relate;
(2) compute — every declared operator must be runnable Python that the explorer can call;
(3) explain — every hypothesis must be linked to the data structure that realizes it.

Return ONE JSON object with exactly these top-level keys:

- "schema_version": "{schema_version}"
- "candidate_id": "c_<task>_<yourtag>"
- "task_id": "{task_id}"
- "content":
    - "sorts": list of semantic kinds you need (each {{"name", "kind", "description"}}; kind ∈ hypothesis|mechanism|factor|intervention|concept|other)
    - "hypotheses": list {{"id", "sort", "statement", "evidence_links"}} — statements must be scientifically concrete and grounded in the evidence
    - "relations": list {{"name", "type", "signature", "pairs", "semantics"}} — type ∈ similarity|dependency|causality|composition|compatibility|contrast|other; signature lists sort names; pairs reference hypothesis ids; semantics states the scientific meaning a fidelity probe would test
    - "required_distinctions": list {{"classes", "reason"}} — declare which pairs of hypotheses the substrate MUST keep computationally distinct (declaring too few does not excuse you: the audit adds its own)
- "form":
    - "self_description": one honest paragraph on how the substrate is organized computationally
    - "carriers": list {{"name", "structure", "payload"}} — the actual data objects (payload holds the data or the code that generates it)
    - "operators": list {{"name", "signature", "semantics", "code"}} — each is a complete Python function (def ...); inputs as declared names/types; semantics states exactly what it computes; it will be executed in a sandbox and checked against its semantics
    - "exploration_interfaces": list {{"name", "operator", "meaning_assumption"}} — the queries an explorer would run; each must bind a declared operator
- "realization":
    - "pairs": list {{"hyp_id", "carrier_ref", "note"}} linking each hypothesis to the carrier(s) that encode it (one-to-many and partial allowed)
    - "coverage_claim": how the pairs cover every required_distinction
- "interp":
    - "rule_description": how a NEW natural-language hypothesis would be mapped into the hypothesis set
    - "set_valued": true (the mapping may return a set; an empty result and an unresolved case are different outcomes)
    - "ambiguity_policy": how genuine ambiguity is handled — split into separately explorable branches, never collapsed into one object
- "claims": list {{"text", "kind"}} — kind ∈ scientific|mathematical|computational; everything you assert must be declared here
- "provenance": {{"proposer_model": "{actor}", "arm": "{arm}", "decode_seed": {decode_seed}, "prompt_hash": "{prompt_hash}", "created_utc": "{created_utc}"}}

Hard rules:
- every operator's code defines its function and runs in ≤30s with no network;
- operator code must not know its probe inputs in advance: each operator receives
  the objects declared in your own carriers (e.g. the payload you put there) and
  must handle those objects exactly — if a carrier holds a table-like payload,
  the operator must accept that same table object, not something of another shape;
- use canonical JSON encodings: relation "pairs" are lists of [id_a, id_b] lists;
  required_distinctions entries are objects {{"classes": [[id_a, id_b], ...],
  "reason": str}}, never bare lists; realization
  "carrier_ref" is a single carrier name string (or a list of such strings);
  every referenced hypothesis id must match a declared hypothesis id exactly;
- stdlib (json, math, statistics, random, datetime, collections, itertools, csv)
  and pandas/numpy/scipy are available to operator code; no network, no file I/O;
- every sort/hypothesis/carrier referenced anywhere must be declared;
- "set_valued" must be true;
- do not include any keys other than those listed; output ONLY the JSON object.

=== TASK ===
{task_render}
"""


class MenuWordViolation(AssertionError):
    pass


def _ci_guard() -> None:
    """Fail-fast CI: the template must not hint any representational family
    and must remain .format()-able (literal braces doubled). A format error
    previously killed every bench lane at first propose (KeyError)."""
    hits = check_menu_words(PROMPT_TEMPLATE)
    if hits:
        raise MenuWordViolation(f"PROMPT_TEMPLATE contains menu words: {hits}")
    try:
        PROMPT_TEMPLATE.format(
            schema_version=SCHEMA_VERSION, task_id="ci", task_render="ci",
            actor="ci", arm="ci", decode_seed=0, prompt_hash="ci",
            created_utc="ci")
    except (KeyError, IndexError, ValueError) as e:
        raise MenuWordViolation(f"PROMPT_TEMPLATE not .format()-able: {e}") from e


_ci_guard()  # import-time enforcement


def render_task(task: dict) -> str:
    """Deterministic task render. Task authors' text is CI-checked upstream
    (bench/openform_bench.py validates each frozen task's render text)."""
    parts = [f"TASK {task.get('task_id', '?')}"]
    for k in ("semantic_description", "evidence", "required_interactions"):
        if task.get(k):
            parts.append(f"## {k}\n{task[k]}")
    return "\n\n".join(parts)


class Proposer:
    def __init__(self, actor: str, base_url: str = ACTOR_URL,
                 api_key: str = API_KEY, timeout: float = 900.0,
                 max_tokens: int = 32768):
        self.actor = actor
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _post(self, prompt: str, seed: int) -> str:
        body = json.dumps({
            "model": self.actor,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "seed": seed,
            "max_tokens": self.max_tokens,
            "enable_thinking": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    obj = json.loads(r.read().decode("utf-8"))
                return obj["choices"][0]["message"]["content"]
            except Exception as e:
                last = e
                time.sleep(min(45.0, 2 ** attempt * 2))
        raise RuntimeError(f"proposer post failed: {last}")

    def propose(self, task: dict, arm: str, decode_seed: int,
                extra_context: str = "") -> tuple:
        """Returns (chi_dict | None, err_str, meta). extra_context carries the
        Fast/Slow feedback text on attempt ≥2 (rendered by fastslow.Feedback)."""
        task_render = render_task(task) + (f"\n\n=== FEEDBACK ===\n{extra_context}"
                                           if extra_context else "")
        # FIX (DEV-20260918-prompt-hash-content): prompt_hash previously hashed
        # PROMPT_TEMPLATE itself - a constant across arms - so provenance could
        # never distinguish a menu prompt from an open prompt and a dead
        # injection was invisible in the data. Hash the full rendered prompt
        # (with hash/timestamp placeholders) so provenance is content-addressed.
        pre = PROMPT_TEMPLATE.format(
            schema_version=SCHEMA_VERSION, task_id=task.get("task_id", "?"),
            task_render=task_render,
            actor=self.actor, arm=arm, decode_seed=decode_seed,
            prompt_hash="PENDING", created_utc="PENDING")
        prompt_hash = "sha256:" + _hex(_hashlibsha(pre))
        prompt = PROMPT_TEMPLATE.format(
            schema_version=SCHEMA_VERSION, task_id=task.get("task_id", "?"),
            task_render=task_render,
            actor=self.actor, arm=arm, decode_seed=decode_seed,
            prompt_hash=prompt_hash,
            created_utc=_utc())
        raw = self._post(prompt, seed=decode_seed)
        m = _JSON_RE.search(raw)
        if not m:
            return None, "no JSON object in proposer output", {"raw_head": raw[:300]}
        try:
            try:
                chi = json.loads(m.group(0))
            except json.JSONDecodeError:
                # models emit raw newlines/tabs inside string values; strip
                # control chars inside string literals before failing the cell
                chi = json.loads(_escape_control_chars(m.group(0)))
        except json.JSONDecodeError as e:
            return None, f"invalid JSON: {e}", {"raw_head": raw[:300]}
        res: ValidationResult = validate(chi)
        if not res.ok:
            return chi, "; ".join(f"{e.code}:{e.detail[:80]}" for e in res.errors), {
                "substrate_hash": res.substrate_hash, "schema_fail": True}
        return chi, "", {"substrate_hash": res.substrate_hash}


# ---- tiny utils (no external deps in this module) ------------------------- #
def _escape_control_chars(s: str) -> str:
    """Escape raw control characters that appear inside JSON string literals
    (models commonly emit literal newlines/tabs in long text values)."""
    out, in_str, i = [], False, 0
    while i < len(s):
        c = s[i]
        if in_str:
            if c == "\\" and i + 1 < len(s):
                out.append(s[i:i + 2])
                i += 2
                continue
            if c == '"':
                in_str = False
                out.append(c)
            elif ord(c) < 0x20:
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(c, f"\\u{ord(c):04x}"))
            else:
                out.append(c)
        else:
            if c == '"':
                in_str = True
            out.append(c)
        i += 1
    return "".join(out)


def _hashlibsha(s: str) -> bytes:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).digest()


def _hex(b: bytes) -> str:
    return b.hex()[:16]


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
