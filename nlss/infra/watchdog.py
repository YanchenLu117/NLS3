"""Batch liveness watchdog.

Periodically evaluates, for one batch:

* **heartbeat freshness** — ``--heartbeat``; stale after ``--max-age``
  seconds (default 1800, matching the SOP stop criterion);
* **canary circuit** — ``--canary-state``; alert while open;
* **new non-zero rc lines** — ``--rc-log``; counts lines matching
  ``rc=<nonzero>`` that appeared since the last check (byte offset is
  tracked in state, so the log may be appended concurrently);
* **disk usage** — ``--disk MOUNT=PCT`` (repeatable).

On a new trigger the watchdog raises its own pause flag (owner
``watchdog``) and appends a record to ``watchdog_alerts.jsonl``; when
all checks clear it removes a flag it owns.  Alert transitions are
latched in state so a persistent condition produces one alert line per
transition, not one per poll.

Usage::

    PYTHONPATH=src python -m nlss.infra.watchdog \\
        --heartbeat work/E_sciexplorer/runs/heartbeat.json \\
        --canary-state work/infra_shared/canary_state.json \\
        --rc-log work/E_sciexplorer/runs/batch_main.log \\
        --disk /usr/data=90 \\
        --state work/E_sciexplorer/runs/watchdog_state.json \\
        --pause-file work/infra_shared/watchdog_pause.flag
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ._util import append_jsonl, atomic_write_json, log_line, read_json

RC_RE = re.compile(r"\brc=(\d+)")
FLAG_OWNER = "watchdog"
DEFAULT_INTERVAL_S = 60.0
DEFAULT_MAX_HEARTBEAT_AGE_S = 1800.0


class Watchdog:
    """Liveness monitor for one batch; file-coordinated, non-invasive."""

    def __init__(
        self,
        *,
        state_path: Path,
        alerts_path: Path,
        pause_path: Path,
        heartbeat: Optional[Path] = None,
        max_heartbeat_age_s: float = DEFAULT_MAX_HEARTBEAT_AGE_S,
        canary_state: Optional[Path] = None,
        rc_log: Optional[Path] = None,
        disk_checks: Optional[Dict[str, float]] = None,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        self.state_path = Path(state_path)
        self.alerts_path = Path(alerts_path)
        self.pause_path = Path(pause_path)
        self.heartbeat = Path(heartbeat) if heartbeat else None
        self.max_heartbeat_age_s = max_heartbeat_age_s
        self.canary_state = Path(canary_state) if canary_state else None
        self.rc_log = Path(rc_log) if rc_log else None
        self.disk_checks = dict(disk_checks or {})
        self.interval_s = interval_s
        self.state = {"rc_offset": 0, "alerts": {}}

    # ------------------------------------------------------------- state
    def load(self) -> None:
        data = read_json(self.state_path) or {}
        self.state = {
            "rc_offset": int(data.get("rc_offset", 0)),
            "alerts": dict(data.get("alerts", {})),
        }

    def save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    # ------------------------------------------------------------- checks
    def _check_heartbeat(self) -> Optional[str]:
        if self.heartbeat is None:
            return None
        data = read_json(self.heartbeat)
        if not isinstance(data, dict) or "ts" not in data:
            return "heartbeat missing or unreadable"
        age = max(0.0, time.time() - float(data["ts"]))
        if age > self.max_heartbeat_age_s:
            return f"heartbeat stale for {age:.0f}s (> {self.max_heartbeat_age_s:.0f}s)"
        return None

    def _check_canary(self) -> Optional[str]:
        if self.canary_state is None:
            return None
        data = read_json(self.canary_state)
        if isinstance(data, dict) and data.get("circuit") == "open":
            updated = time.strftime("%H:%M:%S",
                                    time.localtime(data.get("updated_ts", 0)))
            return f"canary circuit open (updated {updated})"
        return None

    def _check_rc(self) -> Optional[str]:
        """Count new ``rc=<nonzero>`` lines since the last check."""
        if self.rc_log is None or not self.rc_log.exists():
            return None
        size = self.rc_log.stat().st_size
        offset = int(self.state.get("rc_offset", 0))
        if size < offset:  # log rotated/truncated -> rescan from start
            offset = 0
        with self.rc_log.open("r", errors="replace") as fh:
            fh.seek(offset)
            new_text = fh.read()
            end_offset = fh.tell()
        self.state["rc_offset"] = end_offset
        failures = [int(m.group(1)) for m in RC_RE.finditer(new_text)
                    if int(m.group(1)) != 0]
        if failures:
            return f"{len(failures)} new rc!=0 line(s): rc values {sorted(set(failures))}"
        return None

    def _check_disk(self) -> Optional[str]:
        problems = []
        for mount, threshold in sorted(self.disk_checks.items()):
            try:
                usage = shutil.disk_usage(mount)
                pct = 100.0 * usage.used / usage.total
            except OSError as exc:
                problems.append(f"{mount}: unusable ({exc})")
                continue
            if pct > float(threshold):
                problems.append(f"{mount}: {pct:.0f}% > {threshold:.0f}%")
        return "; ".join(problems) if problems else None

    def _evaluate(self) -> Dict[str, str]:
        """Run all configured checks; name -> alert detail (or absent)."""
        alerts: Dict[str, str] = {}
        for name, detail in (
            ("heartbeat", self._check_heartbeat()),
            ("canary", self._check_canary()),
            ("rc_failures", self._check_rc()),
            ("disk", self._check_disk()),
        ):
            if detail:
                alerts[name] = detail
        return alerts

    # -------------------------------------------------------------- flag
    def _flag_owner(self) -> Optional[str]:
        data = read_json(self.pause_path)
        if isinstance(data, dict):
            return data.get("owner")
        return None

    def _reconcile_flag(self, active: Dict[str, str]) -> None:
        if active:
            if not self.pause_path.exists():
                atomic_write_json(self.pause_path, {
                    "owner": FLAG_OWNER,
                    "reason": "; ".join(f"{k}: {v}" for k, v in active.items()),
                    "ts": time.time(),
                })
        else:
            if (self.pause_path.exists()
                    and self._flag_owner() == FLAG_OWNER):
                self.pause_path.unlink(missing_ok=True)

    # -------------------------------------------------------------- loop
    def step(self) -> Dict[str, str]:
        """One evaluation cycle; latches alert transitions to jsonl."""
        active = self._evaluate()
        previous = self.state["alerts"]
        for name, detail in active.items():
            if name not in previous or previous[name] != detail:
                append_jsonl(self.alerts_path, {
                    "schema": 1, "ts": time.time(), "kind": f"watchdog_{name}",
                    "detail": detail,
                })
                log_line(f"watchdog: ALERT {name}: {detail}")
        for name in previous:
            if name not in active:
                append_jsonl(self.alerts_path, {
                    "schema": 1, "ts": time.time(),
                    "kind": f"watchdog_{name}_recovered",
                })
                log_line(f"watchdog: RECOVERED {name}")
        self.state["alerts"] = active
        self._reconcile_flag(active)
        self.save()
        return active

    def _enabled_checks(self) -> List[str]:
        names = []
        if self.heartbeat is not None:
            names.append("heartbeat")
        if self.canary_state is not None:
            names.append("canary")
        if self.rc_log is not None:
            names.append("rc_failures")
        if self.disk_checks:
            names.append("disk")
        return names

    def run(self) -> None:
        self.load()
        log_line(f"watchdog: loop start interval={self.interval_s:.0f}s "
                 f"checks={self._enabled_checks()}")
        while True:
            try:
                self.step()
            except Exception as exc:  # never let the loop die
                log_line(f"watchdog: step error {type(exc).__name__}: {exc}")
            time.sleep(self.interval_s)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="batch liveness watchdog")
    parser.add_argument("--heartbeat", default=None)
    parser.add_argument("--max-age", type=float,
                        default=DEFAULT_MAX_HEARTBEAT_AGE_S)
    parser.add_argument("--canary-state", default=None)
    parser.add_argument("--rc-log", default=None)
    parser.add_argument("--disk", action="append", default=[],
                        metavar="MOUNT=PCT", help="repeatable")
    parser.add_argument("--state", required=True)
    parser.add_argument("--alerts", required=True)
    parser.add_argument("--pause-file", required=True)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    disk_checks: Dict[str, float] = {}
    for item in args.disk:
        mount, _, pct = item.partition("=")
        disk_checks[mount] = float(pct if pct else 90)

    watchdog = Watchdog(
        state_path=Path(args.state), alerts_path=Path(args.alerts),
        pause_path=Path(args.pause_file),
        heartbeat=Path(args.heartbeat) if args.heartbeat else None,
        max_heartbeat_age_s=args.max_age,
        canary_state=Path(args.canary_state) if args.canary_state else None,
        rc_log=Path(args.rc_log) if args.rc_log else None,
        disk_checks=disk_checks, interval_s=args.interval,
    )
    watchdog.load()
    if args.once:
        active = watchdog.step()
        print("watchdog: " + ("OK" if not active else f"ALERTS {active}"))
        return 0 if not active else 1
    watchdog.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
