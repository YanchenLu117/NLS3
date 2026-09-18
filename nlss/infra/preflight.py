"""Pre-launch checklist for scored/confirmatory batches. Exit 0 = green.

Every check is optional and composable; a report json is written when
``--report`` is given.  Hard failures (severity ``fail``) produce a
non-zero exit; warnings (``warn``) do not.

Checks
------
``--registry-dir DIR``
    Every ``*.json`` in DIR that has a ``*.sha256`` sidecar verifies
    against it (sidecar format: ``sha256:<hex>`` or ``<hex>``).
``--decision-registry FILE``
    A DECISION registry json (e.g. DECISION_REGISTRY_V3_20260831.json):
    every ``registries.<name>.file`` must self-pin — the document's own
    ``registry_hash`` equals the DECISION's recorded ``v3_hash`` — and
    must pass the repo's canonical verifier (``nlss.v8.registry``)
    when importable.
``--endpoint URL`` (+ ``--api-key-env`` / ``--api-key-file``)
    One-shot endpoint smoke (models GET; add ``--chat`` for a tiny
    completion that exercises inference).
``--disk MOUNT=PCT``
    Usage above PCT percent fails (default 90). Repeatable.
``--budget FILE``
    json ``{"ceiling": 20, "lines": {"E-line": {"pattern": "...",
    "expect": 10}}}`` — live processes are counted per pattern via ps;
    the hard gate is the ceiling, per-line drift is a warning.  Run on
    the server where the jobs run.

Usage (A-line style, before a confirmatory launch)::

    PYTHONPATH=src python -m nlss.infra.preflight \\
        --registry-dir work/A_hypospace/p0 \\
        --decision-registry work/A_hypospace/p0/DECISION_REGISTRY_V3_20260831.json \\
        --endpoint https://host:port/v1 --api-key-file key.txt \\
        --disk /usr/data=90 --report work/A_hypospace/preflight_report.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ._util import log_line, read_api_key, read_json, sha256_file, strip_sha_prefix
from .canary import probe_chat, probe_models

DEFAULT_DISK_THRESHOLD_PCT = 90.0


@dataclass
class Check:
    """One preflight outcome."""

    name: str
    ok: bool
    detail: str
    severity: str = "fail"  # "fail" gates the launch; "warn" does not


def verify_sidecars(registry_dir: Path) -> List[Check]:
    """Verify every ``*.json`` in dir against its ``*.sha256`` sidecar."""
    checks: List[Check] = []
    json_paths = sorted(Path(registry_dir).glob("*.json"))
    if not json_paths:
        return [Check("registry_dir", False,
                      f"no *.json under {registry_dir}")]
    for json_path in json_paths:
        sidecar = json_path.with_suffix(".sha256")
        if not sidecar.exists():
            continue  # sidecars are optional per file
        try:
            recorded = strip_sha_prefix(sidecar.read_text().split()[0])
        except Exception as exc:
            checks.append(Check(f"sidecar:{json_path.name}", False,
                                f"unreadable sidecar ({exc})"))
            continue
        actual = sha256_file(json_path)
        ok = actual == recorded
        checks.append(Check(
            f"sidecar:{json_path.name}", ok,
            f"{'PASS' if ok else 'FAIL'} recorded={recorded[:16]} "
            f"actual={actual[:16]}"))
    if not checks:
        checks.append(Check("registry_dir", True,
                            f"{len(json_paths)} json file(s), no sidecars"))
    return checks


def verify_decision_registry(decision_path: Path) -> List[Check]:
    """Verify DECISION registry pins against the registry documents.

    Repo convention (src/nlss/v8/registry.py): a registry document is
    *self-hashing* — it carries ``registry_hash`` = sha256 over its
    canonical serialization, and the DECISION records that same value
    as ``v3_hash``.  The check therefore pins DECISION <-> document
    self-hash, and additionally re-runs the repo's own canonical
    verifier when importable.  (Raw file bytes are covered separately
    by the ``*.sha256`` sidecar check.)
    """
    data = read_json(decision_path)
    registries = (data or {}).get("registries") or {}
    if not registries:
        return [Check("decision_registry", False,
                      f"{decision_path} has no registries table")]
    try:
        from nlss.v8.registry import verify_registry_hash as _verify
    except Exception:
        _verify = None  # type: ignore[assignment]
    checks: List[Check] = []
    base = Path(decision_path).parent
    for name in sorted(registries):
        spec = registries[name]
        target = base / spec.get("file", "")
        recorded = str(spec.get("v3_hash", "")).replace("sha256:", "")
        document = read_json(target)
        if not target.exists() or not isinstance(document, dict):
            checks.append(Check(f"decision:{name}", False,
                                f"missing or unreadable {target}"))
            continue
        self_hash = str(document.get("registry_hash", "")
                        ).replace("sha256:", "")
        pin_ok = bool(recorded) and self_hash == recorded
        if _verify is None:
            internal_ok, internal_detail = None, "verifier unavailable"
        else:
            try:
                _verify(document)
                internal_ok, internal_detail = True, "self-hash verified"
            except Exception as exc:  # RegistryHashMismatch et al.
                internal_ok, internal_detail = False, f"{exc}"
        ok = pin_ok and internal_ok is not False
        checks.append(Check(
            f"decision:{name}", ok,
            f"{'PASS' if ok else 'FAIL'} pin={'match' if pin_ok else 'MISMATCH'} "
            f"internal={internal_detail}"))
    return checks


def endpoint_smoke(endpoint: str, api_key: str, model: str,
                   chat: bool = False) -> Check:
    if chat:
        probe = probe_chat(endpoint, api_key, model)
        kind = "chat"
    else:
        probe = probe_models(endpoint, api_key)
        kind = "models"
    return Check(
        f"endpoint_{kind}", probe.ok,
        f"{'PASS' if probe.ok else 'FAIL'} latency={probe.latency_s:.2f}s "
        f"detail={probe.detail}")


def disk_check(mount: str,
               threshold_pct: float = DEFAULT_DISK_THRESHOLD_PCT) -> Check:
    try:
        usage = shutil.disk_usage(mount)
        pct = 100.0 * usage.used / usage.total
    except OSError as exc:
        return Check(f"disk:{mount}", False, f"unusable mount ({exc})")
    ok = pct <= threshold_pct
    return Check(f"disk:{mount}", ok,
                 f"{'PASS' if ok else 'FAIL'} {pct:.0f}% used "
                 f"(threshold {threshold_pct:.0f}%)")


def count_processes(pattern: str) -> int:
    """Count live processes whose command line contains ``pattern``.

    Never counts the preflight tooling itself (any ``nlss.infra``
    process), so patterns do not need quoting tricks.
    """
    result = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                            text=True)
    my_pid = str(__import__("os").getpid())
    count = 0
    for line in result.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid, args = parts
        if pid == my_pid or "nlss.infra" in args:
            continue
        if pattern in args:
            count += 1
    return count


def budget_check(budget: Dict,
                 counter: Callable[[str], int] = count_processes
                 ) -> List[Check]:
    """Gate: total live endpoint jobs <= ceiling; drift is a warning."""
    ceiling = int(budget.get("ceiling", 0))
    total = 0
    checks: List[Check] = []
    for name in sorted(budget.get("lines", {})):
        spec = budget["lines"][name]
        live = counter(spec.get("pattern", ""))
        total += live
        expect = spec.get("expect")
        if expect is not None and live != int(expect):
            checks.append(Check(
                f"budget:{name}", True,
                f"drift: live={live} expect={expect}", severity="warn"))
    checks.append(Check(
        "budget:ceiling", total <= ceiling,
        f"{'PASS' if total <= ceiling else 'FAIL'} total={total} "
        f"ceiling={ceiling}"))
    return checks


def run_preflight_checks(
    *,
    registry_dir: Optional[Path] = None,
    decision_registry: Optional[Path] = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint_chat: bool = False,
    endpoint_model: str = os.environ.get("NLSS_LLM_MODEL", ""),
    disks: Optional[Dict[str, float]] = None,
    budget: Optional[Dict] = None,
) -> List[Check]:
    """Compose all requested checks into one list."""
    checks: List[Check] = []
    if registry_dir:
        checks.extend(verify_sidecars(Path(registry_dir)))
    if decision_registry:
        checks.extend(verify_decision_registry(Path(decision_registry)))
    if endpoint and api_key:
        checks.append(endpoint_smoke(endpoint, api_key, endpoint_model,
                                     chat=endpoint_chat))
    for mount, pct in sorted((disks or {}).items()):
        checks.append(disk_check(mount, pct))
    if budget:
        checks.extend(budget_check(budget))
    return checks


def _print_report(checks: List[Check]) -> bool:
    all_ok = True
    for check in checks:
        marker = "PASS" if check.ok else ("WARN" if check.severity == "warn"
                                          else "FAIL")
        if not check.ok and check.severity == "fail":
            all_ok = False
        print(f"[preflight:{marker}] {check.name}: {check.detail}")
    return all_ok


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="pre-launch checklist")
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument("--decision-registry", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--endpoint-model", default=os.environ.get("NLSS_LLM_MODEL", ""))
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--chat", action="store_true",
                        help="smoke with a tiny completion (default: models)")
    parser.add_argument("--disk", action="append", default=[],
                        metavar="MOUNT=PCT")
    parser.add_argument("--budget", default=None,
                        help="budget json file (see module docstring)")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    if not any([args.registry_dir, args.decision_registry, args.endpoint,
                args.disk, args.budget]):
        parser.error("nothing to check: pass at least one check option")

    api_key = None
    if args.endpoint:
        api_key = read_api_key(args.api_key_env, args.api_key_file)

    disks: Dict[str, float] = {}
    for item in args.disk:
        mount, _, pct = item.partition("=")
        disks[mount] = float(pct if pct else DEFAULT_DISK_THRESHOLD_PCT)

    budget = read_json(args.budget) if args.budget else None

    checks = run_preflight_checks(
        registry_dir=args.registry_dir,
        decision_registry=args.decision_registry,
        endpoint=args.endpoint, api_key=api_key,
        endpoint_chat=args.chat, endpoint_model=args.endpoint_model,
        disks=disks, budget=budget,
    )
    all_ok = _print_report(checks)
    if args.report:
        from ._util import atomic_write_json

        atomic_write_json(args.report, {
            "schema": 1, "ts": time.time(), "all_ok": all_ok,
            "checks": [check.__dict__ for check in checks],
        })
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
