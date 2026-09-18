"""Endpoint health canary with a rolling-window circuit breaker.

Repeatedly probes the model endpoint and maintains a small JSON state
file.  When failures accumulate past the trip threshold the circuit
OPENS and a pause flag file appears; endpoint-consuming batches check
that flag before dispatching new jobs (a one-line bash test) and resume
when it disappears.  This makes queue-burning endpoint incidents — such
as the 2026-08-31 gateway outage that killed 139 E-line cells —
structurally impossible: new work stops while per-job retry budgets
absorb transient errors, and everything resumes automatically on
recovery.

The flag file is bash-compatible (existence == open) and ownership-aware:
the canary only removes a flag it created, so a manual ``touch flag`` by
an engineer is never silently undone.

State file schema (``canary_state.json``)::

    {"schema": 1, "endpoint": "...", "circuit": "open" | "closed",
     "updated_ts": 1790000000.0,
     "probes": [{"ts": ..., "ok": true, "latency_s": 0.32,
                 "detail": "http 200"}]}

Usage (foreground, nohup, or under nlss.infra.supervisor)::

    PYTHONPATH=src python -m nlss.infra.canary \\
        --endpoint https://host:port/v1 --api-key-file key.txt \\
        --state work/infra_shared/canary_state.json \\
        --pause-file work/infra_shared/canary_pause.flag

Bash integration for a worker loop::

    while true; do
        [ -f work/infra_shared/canary_pause.flag ] && { sleep 60; continue; }
        dispatch_next_job
    done
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from ._util import atomic_write_json, log_line, read_api_key, read_json

DEFAULT_INTERVAL_S = 300.0   # SOP: one probe per 5 minutes
DEFAULT_WINDOW = 6           # rolling window of probes (~30 min)
DEFAULT_TRIP_FAILURES = 3    # >= this many failures in window -> OPEN
DEFAULT_CONSECUTIVE_FAIL = 2  # ... or this many failures in a row -> OPEN
DEFAULT_RESET_LOOKBACK = 3   # CLOSED again when the last N probes are ok
DEFAULT_PROBE_TIMEOUT_S = 60.0
FLAG_OWNER = "canary"


@dataclass
class Probe:
    """One endpoint probe outcome."""

    ok: bool
    latency_s: float
    detail: str = ""


def probe_chat(
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
    max_tokens: int = 16,
) -> Probe:
    """Tiny chat completion — exercises the full inference path."""
    url = endpoint.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "ping"}],
         "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    return _do_request(req, timeout)


def probe_models(endpoint: str, api_key: str, timeout: float = 30.0) -> Probe:
    """GET /models — zero-token liveness probe."""
    url = endpoint.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    return _do_request(req, timeout)


def _do_request(req: urllib.request.Request, timeout: float) -> Probe:
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Probe(resp.status == 200, time.monotonic() - t0,
                         f"http {resp.status}")
    except urllib.error.HTTPError as exc:
        return Probe(False, time.monotonic() - t0, f"http {exc.code}")
    except Exception as exc:  # any transport error is a failed probe
        return Probe(False, time.monotonic() - t0,
                     f"{type(exc).__name__}: {exc}"[:200])


class Canary:
    """Rolling-window circuit breaker over an endpoint probe."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        state_path: Path,
        pause_path: Path,
        probe_fn: Optional[Callable[[], Probe]] = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        window: int = DEFAULT_WINDOW,
        trip_failures: int = DEFAULT_TRIP_FAILURES,
        consecutive_fail: int = DEFAULT_CONSECUTIVE_FAIL,
        reset_lookback: int = DEFAULT_RESET_LOOKBACK,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.state_path = Path(state_path)
        self.pause_path = Path(pause_path)
        self.probe_fn: Callable[[], Probe] = probe_fn or (
            lambda: probe_chat(endpoint, api_key, model)
        )
        self.interval_s = interval_s
        self.window = window
        self.trip_failures = trip_failures
        self.consecutive_fail = consecutive_fail
        self.reset_lookback = reset_lookback
        self.probes: List[dict] = []
        self.circuit = "closed"

    # ------------------------------------------------------------- state
    def load(self) -> None:
        data = read_json(self.state_path) or {}
        self.probes = list(data.get("probes", []))[-self.window:]
        self.circuit = data.get("circuit", "closed")

    def save(self) -> None:
        atomic_write_json(self.state_path, {
            "schema": 1,
            "endpoint": self.endpoint,
            "circuit": self.circuit,
            "updated_ts": time.time(),
            "probes": self.probes[-self.window:],
        })

    # ------------------------------------------------------------- rules
    def record(self, probe: Probe) -> None:
        self.probes.append({
            "ts": time.time(),
            "ok": bool(probe.ok),
            "latency_s": round(probe.latency_s, 3),
            "detail": probe.detail,
        })
        self.probes = self.probes[-self.window:]

    def _failures_in_window(self) -> int:
        return sum(1 for p in self.probes if not p["ok"])

    def _consecutive_failures(self) -> int:
        count = 0
        for probe in reversed(self.probes):
            if probe["ok"]:
                break
            count += 1
        return count

    def _last_n_ok(self, n: int) -> bool:
        return len(self.probes) >= n and all(p["ok"] for p in self.probes[-n:])

    def decide(self) -> str:
        """New circuit state after applying trip / reset rules."""
        if self._consecutive_failures() >= self.consecutive_fail:
            return "open"
        if (len(self.probes) >= self.window
                and self._failures_in_window() >= self.trip_failures):
            return "open"
        if self.circuit == "open" and self._last_n_ok(self.reset_lookback):
            return "closed"
        return self.circuit

    # -------------------------------------------------------------- flag
    def _flag_owner(self) -> Optional[str]:
        """Owner recorded in the pause flag, or None (missing/unowned)."""
        data = read_json(self.pause_path)
        if isinstance(data, dict):
            return data.get("owner")
        return None  # missing, empty (bash ``touch``), or unparseable

    def apply(self) -> None:
        """Persist state and reconcile the pause flag with ownership."""
        self.save()
        if self.circuit == "open":
            if not self.pause_path.exists():
                atomic_write_json(self.pause_path, {
                    "owner": FLAG_OWNER,
                    "reason": "endpoint error rate over trip threshold",
                    "ts": time.time(),
                })
        else:
            if (self.pause_path.exists()
                    and self._flag_owner() == FLAG_OWNER):
                self.pause_path.unlink(missing_ok=True)

    # -------------------------------------------------------------- loop
    def step(self) -> Probe:
        """One probe -> record -> decide -> apply cycle."""
        probe = self.probe_fn()
        self.record(probe)
        previous = self.circuit
        self.circuit = self.decide()
        self.apply()
        if self.circuit != previous:
            log_line(f"canary: circuit {previous} -> {self.circuit} "
                     f"(probe ok={probe.ok} detail={probe.detail})")
        return probe

    def run(self) -> None:
        self.load()
        log_line(f"canary: loop start endpoint={self.endpoint} "
                 f"interval={self.interval_s:.0f}s window={self.window}")
        while True:
            try:
                probe = self.step()
                log_line(f"canary: ok={probe.ok} "
                         f"latency={probe.latency_s:.2f}s "
                         f"circuit={self.circuit} detail={probe.detail}")
            except Exception as exc:  # never let the loop die
                log_line(f"canary: step error {type(exc).__name__}: {exc}")
            time.sleep(self.interval_s)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="endpoint canary with circuit breaker")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default=os.environ.get("NLSS_LLM_MODEL", ""))
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--state", required=True)
    parser.add_argument("--pause-file", required=True)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--probe", choices=["chat", "models"], default="chat",
                        help="chat exercises inference (default); models is "
                             "zero-token liveness")
    parser.add_argument("--timeout", type=float, default=DEFAULT_PROBE_TIMEOUT_S)
    parser.add_argument("--once", action="store_true",
                        help="single probe then exit (smoke mode)")
    args = parser.parse_args(argv)

    api_key = read_api_key(args.api_key_env, args.api_key_file)
    if args.probe == "chat":
        probe_fn = lambda: probe_chat(  # noqa: E731
            args.endpoint, api_key, args.model, timeout=args.timeout)
    else:
        probe_fn = lambda: probe_models(  # noqa: E731
            args.endpoint, api_key, timeout=args.timeout)

    canary = Canary(
        endpoint=args.endpoint, api_key=api_key, model=args.model,
        state_path=Path(args.state), pause_path=Path(args.pause_file),
        probe_fn=probe_fn, interval_s=args.interval, window=args.window,
    )
    if args.once:
        probe = canary.step()
        print(json.dumps({
            "ok": probe.ok, "latency_s": round(probe.latency_s, 3),
            "detail": probe.detail, "circuit": canary.circuit,
        }))
        return 0 if probe.ok else 1
    canary.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
