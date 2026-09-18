"""op_executor.py — operator execution inside the frozen filesystem sandbox.

Each declared operator is executed inside the frozen jail
(: chroot whitelist + seccomp network-deny, sandbox API:
JailSpec(inputs={name: host_path}, jail_root, timeout_s) + run_in_jail(script,
spec) -> {"returncode", "stdout", "stderr", ...}; script lands at
/work/input/<name>, PYTHONPATH=/work/input).

IMPORTANT (verified 2026-09-17): run_in_jail launches the script via
runpy.run_path, which runs the child with CLOSED stdin. The runner therefore
reads its spec from /work/input/op_spec.json — never from stdin.

This is the exec half of audit fidelity: q_exec compares operator behavior
against its declared `semantics` via probes. Contract:

  run_operator(op_code, entry_candidates, probe_args, timeout_s=30)
      -> {"ok": bool, "result": ..., "error": str, "wall_s": float}

Formal runs MUST use the jail path (use_jail=True, the default).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

SANDBOX_REPO = Path(os.environ.get("NLSS_SANDBOX_REPO", ""))
JAIL_ROOT = Path("/tmp/nlss_jail")  # frozen active dependency
DEFAULT_TIMEOUT_S = 30.0

_RUNNER_SOURCE = r'''
import json, sys

def _safe(x):
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)

# stdin is closed under runpy launch inside the jail; read spec from the
# whitelisted input copy instead.
spec = json.loads(open("/work/input/op_spec.json").read())
ns = {"_safe": _safe}
exec(spec["code"], ns)
fn = None
for name in spec.get("entry_candidates", []):
    fn = ns.get(name)
    if callable(fn):
        break
if fn is None:
    fn = next((v for v in ns.values() if callable(v)), None)
if fn is None:
    print(json.dumps({"ok": False, "error": "no callable entry function found"}))
    sys.exit(0)
try:
    out = fn(*spec["args"])
    print(json.dumps({"ok": True, "result": _safe(out)}))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
'''


def run_operator(op_code: str, entry_candidates: list, probe_args: list,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 use_jail: bool = True) -> dict:
    """Execute one operator probe. Returns {ok, result, error, wall_s}."""
    spec_json = json.dumps({"code": op_code, "entry_candidates": entry_candidates,
                            "args": probe_args}, ensure_ascii=False)
    t0 = time.time()
    if not use_jail:
        # dev-only fallback (pilot debugging on non-root boxes); formal runs
        # must use the jail path.
        try:
            proc = subprocess.run(
                ["python3", "-I", "-c", _RUNNER_SOURCE.replace(
                    'open("/work/input/op_spec.json")', 'sys.stdin')],
                input=spec_json, capture_output=True, text=True,
                timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout>{timeout_s}s",
                    "wall_s": round(time.time() - t0, 2)}
        return _collect(proc, t0)

    # ---- jail path (sandbox exact API) ----
    import sys as _sys
    if str(SANDBOX_REPO) not in _sys.path:
        _sys.path.insert(0, str(SANDBOX_REPO))
    try:
        from nlss_exp.sandbox.fs_jail import JailSpec, run_in_jail  # type: ignore
    except Exception as e:
        return {"ok": False, "error": f"fs_jail import failed: {e}",
                "wall_s": round(time.time() - t0, 2)}

    with tempfile.TemporaryDirectory(prefix="nlss_op_") as td:
        host_spec = Path(td) / "op_spec.json"
        host_spec.write_text(spec_json, encoding="utf-8")
        runner = Path(td) / "op_runner.py"
        runner.write_text(_RUNNER_SOURCE, encoding="utf-8")
        jail_spec = JailSpec(
            inputs={"op_runner.py": runner, "op_spec.json": host_spec},
            jail_root=JAIL_ROOT, timeout_s=int(timeout_s) + 10)
        try:
            res = run_in_jail(runner, jail_spec)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout>{timeout_s}s (jail)",
                    "wall_s": round(time.time() - t0, 2)}
        except Exception as e:
            return {"ok": False, "error": f"jail exec failed: {type(e).__name__}: {e}",
                    "wall_s": round(time.time() - t0, 2)}
        proc = subprocess.CompletedProcess([], res.get("returncode", -1),
                                           res.get("stdout", ""), res.get("stderr", ""))
    return _collect(proc, t0)


def _collect(proc: subprocess.CompletedProcess, t0: float) -> dict:
    wall = round(time.time() - t0, 2)
    if proc.returncode != 0:
        return {"ok": False, "error": f"rc={proc.returncode} {(proc.stderr or '')[-200:]}",
                "wall_s": wall}
    try:
        obj = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": f"unparseable runner output: {(proc.stdout or '')[:200]}",
                "wall_s": wall}
    obj["wall_s"] = wall
    return obj
