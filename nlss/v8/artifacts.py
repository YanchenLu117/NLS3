"""V8 P0-e — artifact standard and evaluator isolation (Detail §15).

Every run writes the common manifest, evidence/proposal/oracle traces, raw
model output, native persistent-state snapshots, metrics, resources, retrieval
log, adapter audit, and an explicit N/A record for non-applicable artifacts.
NLSS runs additionally write representation/grounding/field/controller/revision
artifacts.  The evaluator owns a SEPARATE tree with a signed digest manifest and
an append-only access log; it is joined to the run manifest only after campaign
termination (Detail §15: evaluator isolation).

All JSON payloads are validated against per-kind versioned schemas before the
writer touches disk; ``artifact_manifest.sha256.json`` content-addresses every
actor artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Artifact tree specification (Detail §15) — {round} and {version}/{event_id}
# are substituted by the writer; kinds ending in _jsonl take lists of records.
# ---------------------------------------------------------------------------

ACTOR_TREE: tuple[tuple[str, str], ...] = (
    ("run_manifest", "run_manifest.json"),
    ("artifact_manifest", "artifact_manifest.sha256.json"),
    ("evidence_ledger", "evidence/ledger.jsonl"),
    ("native_state", "native/state_round_{round}.json"),
    ("native_raw_output", "native/raw_output_round_{round}.jsonl"),
    ("decision_trace", "decision/trace_round_{round}.jsonl"),
    ("state_query_questions", "state_query/questions.json"),
    ("state_query_answers", "state_query/answers_round_{round}.jsonl"),
    ("state_query_scores", "state_query/scores_round_{round}.json"),
    ("retrieval_log", "retrieval/queries_and_results.jsonl"),
    ("representation_generators", "representation/generators_v{version}.json"),
    ("representation_relations", "representation/relations_v{version}.json"),
    ("representation_covers", "representation/covers_v{version}.json"),
    ("representation_contract", "representation/contract_v{version}.json"),
    ("representation_certificate", "representation/certificate_v{version}.json"),
    ("representation_completeness_status", "representation/completeness_status_v{version}.json"),
    ("representation_logic_report", "representation/logic_report_v{version}.json"),
    ("representation_fidelity_report", "representation/fidelity_report_v{version}.json"),
    ("audit_actor_relation", "audit/actor_relation_audit_v{version}.json"),
    ("audit_actor_cover", "audit/actor_cover_audit_v{version}.json"),
    ("audit_actor_overlap", "audit/actor_overlap_audit_v{version}.json"),
    ("grounding_generator_map", "grounding/generator_map_v{version}.json"),
    ("grounding_gamma", "grounding/gamma_round_{round}.jsonl"),
    ("field_posterior", "field/posterior_round_{round}.json"),
    ("field_calibration", "field/calibration_round_{round}.json"),
    ("readout_natural_language", "readout/natural_language_round_{round}.md"),
    ("readout_computable", "readout/computable_round_{round}.json"),
    ("controller_decision", "controller/decision_round_{round}.json"),
    ("revision_trigger", "revision/trigger_round_{round}_{event_id}.json"),
    ("revision_diff", "revision/diff_round_{round}_{event_id}.json"),
    ("revision_verdict", "revision/verdict_round_{round}_{event_id}.json"),
    ("revision_migration", "revision/migration_round_{round}_{event_id}.json"),
    ("proposals_candidates", "proposals/candidates_round_{round}.jsonl"),
    ("oracle_observations", "oracle/observations_round_{round}.jsonl"),
    ("metrics_round", "metrics/round_{round}.json"),
    ("resources_usage", "resources/usage_round_{round}.json"),
    ("adapters_identity_audit", "adapters/identity_audit.json"),
    ("adapters_tool_trace", "adapters/tool_trace.jsonl"),
    ("not_applicable_record", "adapters/not_applicable.json"),
)

EVALUATOR_TREE: tuple[tuple[str, str], ...] = (
    ("evaluator_semantic_signatures", "evaluator/semantic_signatures_v{version}.json"),
    ("evaluator_canonical_basis", "evaluator/canonical_basis_v{version}.json"),
    ("evaluator_hidden_relations", "evaluator/hidden_relations_v{version}.json"),
    ("evaluator_relation_audit", "evaluator/relation_audit_v{version}.json"),
    ("evaluator_cover_audit", "evaluator/cover_audit_v{version}.json"),
    ("evaluator_certification_level", "evaluator/certification_level_v{version}.json"),
)

_TEMPLATES = dict(ACTOR_TREE)


class ArtifactSchemaError(ValueError):
    """Payload does not validate against its per-kind versioned schema."""


class EvaluatorSealedError(RuntimeError):
    """Evaluator vault contents accessed before campaign termination."""


# Minimal per-kind versioned schemas (required keys).  §15: every JSON validates.
SCHEMA_VERSION = 1

SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "metrics_round": ("round", "primary_endpoint", "primary_value"),
    "resources_usage": ("round", "llm_calls", "tokens_in", "tokens_out"),
    "controller_decision": ("round", "scores", "batch"),
    "revision_trigger": ("round", "event_id", "trigger"),
    "oracle_observations": ("round", "observations"),
    "state_query_scores": ("round", "macro_accuracy"),
    "field_calibration": ("round", "calibration_method"),
    "representation_contract": ("version", "contract"),
    "evaluator_certification_level": ("version", "achieved_level"),
}


def _validate_payload(kind: str, payload: Mapping[str, Any]) -> None:
    required = SCHEMAS.get(kind)
    if required is None:
        return
    missing = [key for key in required if key not in payload]
    if missing:
        raise ArtifactSchemaError(f"{kind}: missing required keys: {', '.join(missing)} (schema v{SCHEMA_VERSION})")


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvidenceLedger:
    """Append-only evidence ledger D_t (Detail §3.2/§3.13).

    Records are appended (never rewritten); every record carries a content
    fingerprint; ``extends`` verifies D_t ⊆ D_{t+1} by fingerprint prefix —
    evidence persists, only the representation evolves."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, records: list[Mapping[str, Any]]) -> int:
        for record in records:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(_canonical_json(record).decode("utf-8") + "\n")
        return len(records)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def fingerprints(self) -> list[str]:
        return [hashlib.sha256(_canonical_json(r)).hexdigest() for r in self.records()]

    def extends(self, previous_fingerprints: list[str]) -> bool:
        """D_prev ⊆ D_now as a fingerprint prefix (monotone evidence growth)."""
        return self.fingerprints()[: len(previous_fingerprints)] == previous_fingerprints


@dataclass
class RunArtifactWriter:
    """Writes actor artifacts into a run directory and content-addresses them."""

    run_dir: Path
    written: dict[str, str] = field(default_factory=dict)  # kind -> relative path

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, kind: str, *, round_t: int | None = None, version: int | None = None, event_id: str | None = None) -> str:
        if kind not in _TEMPLATES:
            raise KeyError(f"unknown artifact kind: {kind}")
        if round_t is not None and not isinstance(round_t, int):
            raise ArtifactSchemaError(f"{kind}: round_t must be an int")
        if version is not None and not isinstance(version, int):
            raise ArtifactSchemaError(f"{kind}: version must be an int")
        if event_id is not None:
            import re

            if not isinstance(event_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", event_id):
                raise ArtifactSchemaError(f"{kind}: event_id must match [A-Za-z0-9_-]{{1,64}} (path-safety)")
        template = _TEMPLATES[kind]
        placeholders = {p.split("}")[0] for p in template.split("{")[1:]} if "{" in template else set()
        values = {"round": round_t, "version": version, "event_id": event_id}
        for name in placeholders:
            if values[name] is None:
                raise ArtifactSchemaError(
                    f"{kind}: template requires '{name}' — silent placeholder artifacts violate §15 naming"
                )
        return template.format(
            round=values["round"],
            version=values["version"],
            event_id=values["event_id"],
        )

    def write(
        self,
        kind: str,
        payload: Any,
        *,
        round_t: int | None = None,
        version: int | None = None,
        event_id: str | None = None,
    ) -> Path:
        rel = self.resolve_path(kind, round_t=round_t, version=version, event_id=event_id)
        path = self.run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".jsonl"):
            if not isinstance(payload, list):
                raise ArtifactSchemaError(f"{kind}: jsonl payload must be a list of records")
            for record in payload:
                if isinstance(record, Mapping):
                    _validate_payload(kind, record)
            lines = "".join(_canonical_json(r).decode("utf-8") + "\n" for r in payload)
            path.write_text(lines, encoding="utf-8")
        elif rel.endswith(".md"):
            if not isinstance(payload, str):
                raise ArtifactSchemaError(f"{kind}: md payload must be a string")
            path.write_text(payload, encoding="utf-8")
        else:
            if isinstance(payload, Mapping):
                _validate_payload(kind, payload)
            path.write_text(_canonical_json(payload).decode("utf-8"), encoding="utf-8")
        self.written[kind] = rel
        return path

    def write_manifest(self, run_manifest: Mapping[str, Any]) -> Path:
        """Content-address every actor artifact (Detail §15) and write both manifests."""
        entries: dict[str, str] = {}
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name not in ("artifact_manifest.sha256.json", "run_manifest.json"):
                entries[str(path.relative_to(self.run_dir))] = f"sha256:{_sha256_file(path)}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_manifest_hash": f"sha256:{hashlib.sha256(_canonical_json(run_manifest)).hexdigest()}",
            "artifacts": entries,
        }
        am_path = self.run_dir / "artifact_manifest.sha256.json"
        am_path.write_text(_canonical_json(manifest).decode("utf-8"), encoding="utf-8")
        rm_path = self.run_dir / "run_manifest.json"
        rm_path.write_text(_canonical_json(run_manifest).decode("utf-8"), encoding="utf-8")
        return am_path

    def verify(self) -> list[str]:
        """Re-verify every content-addressed artifact; returns problems."""
        am_path = self.run_dir / "artifact_manifest.sha256.json"
        if not am_path.exists():
            return ["artifact_manifest.sha256.json missing"]
        manifest = json.loads(am_path.read_text(encoding="utf-8"))
        problems: list[str] = []
        for rel, recorded in manifest["artifacts"].items():
            path = self.run_dir / rel
            if not path.exists():
                problems.append(f"missing artifact: {rel}")
            elif f"sha256:{_sha256_file(path)}" != recorded:
                problems.append(f"hash mismatch: {rel}")
        return problems


# ------------------------------------------------------------- run manifest

RUN_MANIFEST_MANDATORY: tuple[str, ...] = (
    "p0_registry_hash",
    "commits",
    "environment",
    "model_config_ids",
    "prompts",
    "seed",
    "budgets",
    "thresholds",
    "baseline_roster",
    "result_visibility_time",
    "representation_checkpoint_map",
    "achieved_certification_level",
    "terminal_status",
    "na_reasons",
)


class RunManifestBuilder:
    """run_manifest.json per Detail §15; refuses to emit with unset mandatory fields."""

    def __init__(self) -> None:
        self._fields: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> "RunManifestBuilder":
        if key not in RUN_MANIFEST_MANDATORY:
            raise KeyError(f"unknown run-manifest field: {key}")
        self._fields[key] = value
        return self

    def missing_fields(self) -> list[str]:
        return [k for k in RUN_MANIFEST_MANDATORY if k not in self._fields]

    def build(self) -> dict[str, Any]:
        missing = self.missing_fields()
        if missing:
            raise ArtifactSchemaError(f"run_manifest missing mandatory fields: {', '.join(missing)}")
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **self._fields,
        }


# ---------------------------------------------------------- evaluator vault

EVALUATOR_MANDATORY_DIGEST: tuple[str, ...] = ("certification_level",)


class EvaluatorVault:
    """Evaluator-owned tree: sealed until campaign termination, signed digest,
    append-only access log (Detail §15)."""

    def __init__(self, vault_dir: Path, signing_key: bytes) -> None:
        self.vault_dir = Path(vault_dir)
        self._signing_key = signing_key
        self._sealed = False
        self._templates = dict(EVALUATOR_TREE)
        self._digest: dict[str, Any] | None = None

    def write(self, kind: str, payload: Mapping[str, Any], *, version: int) -> Path:
        if self._sealed:
            raise EvaluatorSealedError("vault sealed: post-seal writes invalidate the digest")
        if kind not in self._templates:
            raise KeyError(f"unknown evaluator artifact kind: {kind}")
        if kind == "evaluator_certification_level":
            _validate_payload(kind, payload)
        rel = self._templates[kind].format(version=version)
        path = self.vault_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_canonical_json(payload).decode("utf-8"), encoding="utf-8")
        return path

    def log_access(self, reader: str, action: str) -> None:
        """Append-only access log: reader identity + timestamp (Detail §15).

        Legal both during and after the campaign; the log lives OUTSIDE the
        digest manifest (R5 review: post-seal reads must not break the
        signature)."""
        entry = {
            "reader": reader,
            "action": action,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        log_path = self.vault_dir / "evaluator" / "access_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(_canonical_json(entry).decode("utf-8") + "\n")

    def seal(self) -> None:
        """Campaign terminated: build the signed digest manifest.

        The append-only access log is deliberately OUTSIDE the digest (post-
        seal reads append legally and must not break the signature)."""
        entries: dict[str, str] = {}
        for path in sorted(self.vault_dir.rglob("*")):
            if path.is_file() and path.name not in ("digest_manifest.json", "access_log.jsonl"):
                entries[str(path.relative_to(self.vault_dir))] = f"sha256:{_sha256_file(path)}"
        digest_payload = {
            "schema_version": SCHEMA_VERSION,
            "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluator_artifacts": entries,
        }
        canonical = _canonical_json(digest_payload)
        signature = hmac.new(self._signing_key, canonical, hashlib.sha256).hexdigest()
        self._digest = {**digest_payload, "signature": f"sha256-hmac:{signature}"}
        (self.vault_dir / "evaluator" / "digest_manifest.json").write_text(
            _canonical_json(self._digest).decode("utf-8"), encoding="utf-8"
        )
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def join_to_run_manifest(self, run_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        """Join the evaluator digest to the run manifest — only post-termination."""
        if not self._sealed or self._digest is None:
            raise EvaluatorSealedError("evaluator vault not sealed: join is forbidden before campaign termination")
        return {**run_manifest, "evaluator_digest_hash": self._digest["evaluator_artifacts"], "evaluator_joined_at": self._digest["sealed_at"]}

    def verify_digest(self) -> list[str]:
        """Verify the ON-DISK digest (auditor semantics): signature over the
        stored payload, then re-hash every current file against it."""
        if not self._sealed:
            raise EvaluatorSealedError("vault not sealed")
        digest_path = self.vault_dir / "evaluator/digest_manifest.json"
        if not digest_path.exists():
            return ["digest_manifest.json missing"]
        stored = json.loads(digest_path.read_text(encoding="utf-8"))
        signature = stored.pop("signature", "")
        expected = hmac.new(self._signing_key, _canonical_json(stored), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, f"sha256-hmac:{expected}"):
            return ["evaluator digest signature mismatch"]
        problems: list[str] = []
        for rel, recorded in stored.get("evaluator_artifacts", {}).items():
            path = self.vault_dir / rel
            if not path.exists():
                problems.append(f"missing evaluator artifact: {rel}")
            elif f"sha256:{_sha256_file(path)}" != recorded:
                problems.append(f"hash mismatch: {rel}")
        return problems
