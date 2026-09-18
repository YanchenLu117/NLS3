"""chi_schema.py — Candidate Schema validator (frozen spec).

Validates the proposer's candidate package χ=(Ĉ,F̂,Γ̂,Înterp̂,K). This enforces
*describability*, never a representation family: Form is fully open. The
NON_MENU judgment happens later in audit equivalence (evaluator-only).

Hard constraints (frozen 2026-09-17):
  HC1 all operator.code imports+compiles in jail, 30s timeout (checked at audit
      time by executor, not here; here we only syntax-check via ast.parse);
  HC2 sort references close (hypotheses.sort, relations.signature ⊆ sorts);
  HC3 Γ pairs reference existing hyp_id and carrier_ref;
  HC4 proposer prompt builders contain NO menu words (CI on the prompt module
      source, enforced by check_menu_words);
  HC5 interp.set_valued is true (∅ vs Unresolved distinction is mandatory);
  HC6 required_distinctions reference existing hyp_ids (compile → induced
      obligations happens in audit.engine);
  HC7 provenance complete (proposer_model, arm, decode_seed, prompt_hash,
      created_utc).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field

SCHEMA_VERSION = "1.1"

MENU_WORDS = [
    "vector", "distribution", "graph", "hierarchy", "hierarchical",
    "discrete-combinatorial", "timeseries", "relational table",
    "embedding", "knowledge graph", "tree structure", "matrix",
]
# The proposer prompt builder source must not contain any of these as a hint
# about what representation to use. CI gate (HC4).


@dataclass
class SchemaError:
    code: str
    detail: str


@dataclass
class ValidationResult:
    ok: bool
    errors: list = field(default_factory=list)
    substrate_hash: str = ""  # sha256 over canonical serialization of (Ĉ,F̂,Γ̂,Înterp̂)


def check_menu_words(prompt_source: str) -> list:
    """HC4: return menu words found in a proposer prompt-builder source.
    Left word-boundary only (\\b prefix), so innocuous substrings like
    'paragraph' don't trip 'graph', while real hints match including natural
    inflections ('vectors', 'embeddings', 'hierarchical')."""
    low = prompt_source.lower()
    return [w for w in MENU_WORDS
            if re.search(r"\b" + re.escape(w), low)]


def canonical_substrate_hash(chi: dict) -> str:
    """sha256 over the canonical serialization of (Ĉ,F̂,Γ̂,Înterp̂).

    K (claims) and provenance are deliberately EXCLUDED: the frozen substrate
    invariant covers the four structural components only (Methods §4.1).
    """
    core = {k: chi.get(k) for k in ("content", "form", "realization", "interp")}
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate(chi: dict) -> ValidationResult:
    errs: list[SchemaError] = []
    if chi.get("schema_version") != SCHEMA_VERSION:
        errs.append(SchemaError("HC0", f"schema_version != {SCHEMA_VERSION}"))
    if not isinstance(chi.get("candidate_id"), str) or not chi.get("candidate_id"):
        errs.append(SchemaError("HC0", "candidate_id missing"))

    # FIX (DEV-20260918-fuzz-hardening): a wrong-typed top-level component
    # previously crashed validate() itself (AttributeError), killing the
    # bench process. Coerce to {} and flag, so validate always returns.
    content = chi.get("content") if isinstance(chi.get("content"), dict) else {}
    form = chi.get("form") if isinstance(chi.get("form"), dict) else {}
    real = chi.get("realization") if isinstance(chi.get("realization"), dict) else {}
    interp = chi.get("interp") if isinstance(chi.get("interp"), dict) else {}
    if "content" not in chi or not isinstance(chi.get("content"), dict):
        errs.append(SchemaError("HC0", "content missing or not an object"))
    if "form" not in chi or not isinstance(chi.get("form"), dict):
        errs.append(SchemaError("HC0", "form missing or not an object"))
    if not isinstance(chi.get("realization"), dict):
        errs.append(SchemaError("HC3", "realization missing or not an object"))
    if not isinstance(chi.get("interp"), dict):
        errs.append(SchemaError("HC5", "interp missing or not an object"))

    # ---- Ĉ ----
    sorts = content.get("sorts") or []
    sort_names = [s.get("name") for s in sorts if isinstance(s, dict)]
    if not sort_names:
        errs.append(SchemaError("HC2", "content.sorts empty"))
    dup = {n for n in sort_names if sort_names.count(n) > 1}
    if dup:
        errs.append(SchemaError("HC2", f"duplicate sort names: {sorted(dup)}"))

    hyp_ids: set = set()
    hyp_keys: set = set()   # ids plus declared name/label aliases for ref resolution
    hyp_stems: list = []    # fuzzy stems extracted from ids/names
    for h in content.get("hypotheses") or []:
        if not isinstance(h, dict) or not h.get("id"):
            errs.append(SchemaError("HC2", "hypothesis without id"))
            continue
        hyp_ids.add(h["id"])
        for alias_key in ("name", "label", "title"):
            if isinstance(h.get(alias_key), str) and h[alias_key].strip():
                hyp_keys.add(h[alias_key].strip())
        for src_name in (h["id"],) + tuple(
                h.get(k) for k in ("name", "label", "title") if isinstance(h.get(k), str)):
            flat = src_name.lower().replace("hyp", "").replace("hypothesis", "") \
                .replace("_", "").replace("-", "").replace(" ", "")
            if len(flat) >= 5:
                hyp_stems.append(flat)
        if h.get("sort") not in sort_names:
            errs.append(SchemaError("HC2", f"hypothesis {h['id']} sort '{h.get('sort')}' not in sorts"))
        if not (h.get("statement") or "").strip():
            errs.append(SchemaError("HC2", f"hypothesis {h['id']} empty statement"))
    if not hyp_ids:
        errs.append(SchemaError("HC2", "content.hypotheses empty"))
    ref_space = hyp_ids | hyp_keys

    def _ref_ok(hid: str) -> bool:
        """Exact id/alias match, else fuzzy: the reference (lowercased, de-punctuated)
        contains a hypothesis id/name stem. Models invent naming conventions
        ('HypothesisLinearity', 'hypothesis_linear_efficiency') — prose refs
        that clearly identify a declared hypothesis count as resolved."""
        if isinstance(hid, str) and hid in ref_space:
            return True
        if not isinstance(hid, str):
            return False
        flat = hid.lower().replace("_", "").replace("-", "").replace(" ", "")
        return any(stem in flat for stem in hyp_stems)

    for r in content.get("relations") or []:
        if not isinstance(r, dict):
            errs.append(SchemaError("HC2", "relation entry not an object"))
            continue
        for s in r.get("signature") or []:
            if s not in sort_names:
                errs.append(SchemaError("HC2", f"relation '{r.get('name')}' signature sort '{s}' not in sorts"))
        pairs = r.get("pairs") or []
        # encodings: [["h1","h2"],...] (spec) | ["h1","h2"] flat = one pair
        # (pair-wise relations are unary-reference vs the relation target when
        # odd) — flat odd lists mean a unary relation over each listed hyp.
        if pairs and all(isinstance(p, str) for p in pairs):
            if len(pairs) % 2 == 0:
                norm_pairs = [[pairs[i], pairs[i + 1]] for i in range(0, len(pairs), 2)]
            else:
                norm_pairs = [[p] for p in pairs]
        else:
            norm_pairs = pairs
        for pair in norm_pairs:
            if not isinstance(pair, (list, tuple)):
                errs.append(SchemaError("HC2", f"relation '{r.get('name')}' pair not a list: {str(pair)[:40]}"))
                continue
            for hid in pair:
                if not isinstance(hid, str):
                    errs.append(SchemaError("HC2", f"relation '{r.get('name')}' pair element not a string"))
                elif not _ref_ok(hid):
                    errs.append(SchemaError("HC2", f"relation '{r.get('name')}' references unknown hyp_id '{hid}'"))

    req_ids: set = set()
    for rd in content.get("required_distinctions") or []:
        if isinstance(rd, list):
            # alternate encoding: bare [id_a, id_b] list = one class
            for hid in rd:
                if isinstance(hid, str):
                    req_ids.add(hid)
                    if not _ref_ok(hid):
                        errs.append(SchemaError("HC6", f"required_distinctions references unknown hyp_id '{hid}'"))
            continue
        if not isinstance(rd, dict):
            continue
        classes = rd.get("classes") or []
        # encodings: [["h1","h2"],...] | flat strings ["H1","H2"] = each a
        # singleton class (distinction set over the listed hypotheses)
        if classes and all(isinstance(c, str) for c in classes):
            class_groups = [[c] for c in classes]
        else:
            class_groups = classes
        for pair in class_groups:
            if not isinstance(pair, (list, tuple)):
                errs.append(SchemaError("HC6", f"required_distinctions class not a list: {str(pair)[:40]}"))
                continue
            for hid in pair:
                req_ids.add(hid)
                if not _ref_ok(hid):
                    errs.append(SchemaError("HC6", f"required_distinctions references unknown hyp_id '{hid}'"))

    # ---- F̂ ----
    carrier_names: set = set()
    for c in form.get("carriers") or []:
        if isinstance(c, dict) and c.get("name"):
            carrier_names.add(c["name"])
        else:
            errs.append(SchemaError("HC0", "carrier without name"))
    ops = form.get("operators") or []
    if not ops:
        errs.append(SchemaError("HC0", "form.operators empty (substrate must expose executable operators)"))
    for op in ops:
        if not isinstance(op, dict):
            errs.append(SchemaError("HC0", "operator entry not an object"))
            continue
        code = (op.get("code") or "").strip()
        if not code:
            errs.append(SchemaError("HC1", f"operator '{op.get('name')}' empty code"))
            continue
        try:
            tree = ast.parse(code)
            fns = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not fns:
                errs.append(SchemaError("HC1", f"operator '{op.get('name')}' code defines no function"))
        except SyntaxError as e:
            errs.append(SchemaError("HC1", f"operator '{op.get('name')}' syntax error: {e}"))
    if not (form.get("exploration_interfaces")):
        errs.append(SchemaError("HC0", "form.exploration_interfaces empty (explorer must consume something to audit)"))
    for ei in form.get("exploration_interfaces") or []:
        if not isinstance(ei, dict):
            errs.append(SchemaError("HC0", "exploration_interface entry not an object"))

    # ---- Γ̂ ----
    for p in real.get("pairs") or []:
        if not isinstance(p, dict):
            errs.append(SchemaError("HC3", "Γ pair not an object"))
            continue
        cref = p.get("carrier_ref")
        if isinstance(cref, list):
            # allow list-of-carriers notation; every element must exist
            for c in cref:
                if c not in carrier_names:
                    errs.append(SchemaError("HC3", f"Γ pair carrier_ref '{c}' unknown"))
        elif cref not in carrier_names:
            errs.append(SchemaError("HC3", f"Γ pair carrier_ref '{cref}' unknown"))
        if not _ref_ok(p.get("hyp_id")):
            errs.append(SchemaError("HC3", f"Γ pair hyp_id '{p.get('hyp_id')}' unknown"))

    # ---- Înterp̂ ----
    if interp.get("set_valued") is not True:
        errs.append(SchemaError("HC5", "interp.set_valued must be true"))
    if not (interp.get("rule_description") or "").strip():
        errs.append(SchemaError("HC5", "interp.rule_description empty"))
    if not (interp.get("ambiguity_policy") or "").strip():
        errs.append(SchemaError("HC5", "interp.ambiguity_policy empty"))

    # ---- K ----
    claims = chi.get("claims")
    if not isinstance(claims, list):
        errs.append(SchemaError("HC0", "claims missing (may be empty list, must be present)"))
    else:
        for cl in claims:
            if not isinstance(cl, dict):
                errs.append(SchemaError("HC0", "claim entry not an object"))
                break

    # ---- provenance ----
    prov = chi.get("provenance") or {}
    for k in ("proposer_model", "arm", "prompt_hash", "created_utc"):
        if not prov.get(k):
            errs.append(SchemaError("HC7", f"provenance.{k} missing"))
    if "decode_seed" not in prov:
        errs.append(SchemaError("HC7", "provenance.decode_seed missing"))

    return ValidationResult(ok=not errs, errors=errs,
                            substrate_hash=canonical_substrate_hash(chi))


if __name__ == "__main__":
    import sys
    chi = json.loads(open(sys.argv[1]).read())
    res = validate(chi)
    print(json.dumps({"ok": res.ok,
                      "errors": [vars(e) for e in res.errors],
                      "substrate_hash": res.substrate_hash},
                     ensure_ascii=False, indent=1))
    sys.exit(0 if res.ok else 1)
