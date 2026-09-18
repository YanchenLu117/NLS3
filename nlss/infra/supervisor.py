"""Restart-on-crash command supervisor with heartbeat and pause support.

Wraps any batch command so that:

* a crashed process is restarted with capped exponential backoff;
* a heartbeat file proves liveness to ``nlss.infra.watchdog`` (touched
  while running, paused, and backing off — the watchdog never sees a
  stale heartbeat caused by the supervisor itself);
* an open canary circuit (pause flag) delays new attempts instead of
  burning them;
* a crash-loop guard gives up after ``max_restarts`` restarts within
  ``restart_window_s`` and writes an alert record instead of
  respawning forever.

Full output goes to per-attempt log files; summary lines (START / END /
rc= / ALERT / WARN / ERROR by default) are additionally appended to a
shared main log.  This replaces the single-mega-log pattern (the direct
motivation was a 951 MB scored_batch.log).

Usage::

    PYTHONPATH=src python -m nlss.infra.supervisor \\
        --heartbeat work/E_sciexplorer/runs/heartbeat.json \\
        --log-dir work/E_sciexplorer/runs/supervisor \\
        --summary-log work/E_sciexplorer/runs/batch_main.log \\
        --pause-file work/infra_shared/canary_pause.flag \\
        -- bash work/E_sciexplorer/v8_scored_batch.sh 10 0
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Pattern, Sequence

from ._util import append_jsonl, atomic_write_json, log_line, read_json

DEFAULT_SUMMARY_RE = re.compile(
    r"\b(?:START|END|HEARTBEAT|ALERT|WARN|ERROR)\b|\brc=\d+")


@dataclass
class SupervisionPolicy:
    """Tuning knobs for :func:`run_under_supervisor`."""

    max_restarts: int = 5          # give up beyond this many restarts ...
    restart_window_s: float = 600.0  # ... within this window (crash loop)
    backoff_init_s: float = 5.0
    backoff_max_s: float = 300.0
    beat_interval_s: float = 30.0  # heartbeat refresh period
    pause_poll_s: float = 15.0     # pause-flag poll period


def touch_heartbeat(path: Path, *, attempt: int, pid: Optional[int],
                    note: str = "running") -> None:
    """Write the liveness heartbeat (atomic JSON)."""
    atomic_write_json(path, {
        "schema": 1, "ts": time.time(), "attempt": attempt,
        "pid": pid, "note": note,
    })


def heartbeat_age_s(path: Path) -> Optional[float]:
    """Seconds since the heartbeat was written (None if unreadable)."""
    data = read_json(path)
    if isinstance(data, dict) and "ts" in data:
        return max(0.0, time.time() - float(data["ts"]))
    return None


def wait_while_paused(pause_file: Path, *, heartbeat: Optional[Path] = None,
                      poll_s: float = 15.0, beat_s: float = 30.0) -> None:
    """Block while the pause flag exists, keeping the heartbeat fresh."""
    last_beat = 0.0
    while Path(pause_file).exists():
        now = time.time()
        if heartbeat is not None and now - last_beat >= beat_s:
            touch_heartbeat(heartbeat, attempt=0, pid=os.getpid(),
                            note="paused: circuit open")
            last_beat = now
        time.sleep(poll_s)


def _sleep_with_heartbeat(delay_s: float, heartbeat: Optional[Path],
                          beat_s: float, note: str) -> None:
    """Sleep while refreshing the heartbeat (backoff windows)."""
    deadline = time.monotonic() + delay_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if heartbeat is not None:
            touch_heartbeat(heartbeat, attempt=0, pid=os.getpid(), note=note)
        time.sleep(min(beat_s, remaining))


def _run_attempt(cmd: Sequence[str], log_path: Path, *,
                 summary_log: Optional[Path] = None,
                 summary_re: Optional[Pattern[str]] = None,
                 heartbeat: Optional[Path] = None, attempt: int = 0,
                 beat_s: float = 30.0, cwd: Optional[str] = None) -> int:
    """Run one attempt; full log per attempt, summary lines to main log."""
    with open(log_path, "w") as full_log:
        proc = subprocess.Popen(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=cwd)
        summary_fh = open(summary_log, "a") if summary_log else None

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                full_log.write(line)
                full_log.flush()
                if summary_fh is not None and (
                        summary_re is None or summary_re.search(line)):
                    summary_fh.write(line)
                    summary_fh.flush()

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        try:
            while proc.poll() is None:
                if heartbeat is not None:
                    touch_heartbeat(heartbeat, attempt=attempt, pid=proc.pid)
                time.sleep(beat_s)
            thread.join(timeout=30)
            return proc.returncode
        finally:
            if summary_fh is not None:
                summary_fh.close()


def run_under_supervisor(cmd: Sequence[str], *, heartbeat: Path,
                         log_dir: Path,
                         policy: Optional[SupervisionPolicy] = None,
                         pause_file: Optional[Path] = None,
                         summary_log: Optional[Path] = None,
                         summary_re: Optional[Pattern[str]] = None,
                         cwd: Optional[str] = None,
                         alert_path: Optional[Path] = None) -> int:
    """Run ``cmd`` under restart/backoff/heartbeat/pause supervision."""
    policy = policy or SupervisionPolicy()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(heartbeat).parent.mkdir(parents=True, exist_ok=True)
    if summary_log is not None:
        Path(summary_log).parent.mkdir(parents=True, exist_ok=True)

    restarts: deque = deque()
    attempt = 0
    while True:
        if pause_file is not None and Path(pause_file).exists():
            log_line(f"supervisor: pause flag present, holding dispatch")
            wait_while_paused(pause_file, heartbeat=heartbeat,
                              poll_s=policy.pause_poll_s,
                              beat_s=policy.beat_interval_s)
        attempt += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"attempt_{attempt:03d}_{stamp}.log"
        log_line(f"supervisor: START attempt={attempt} "
                 f"cmd={' '.join(cmd)} log={log_path.name}")
        rc = _run_attempt(cmd, log_path, summary_log=summary_log,
                          summary_re=summary_re or DEFAULT_SUMMARY_RE,
                          heartbeat=heartbeat, attempt=attempt,
                          beat_s=policy.beat_interval_s, cwd=cwd)
        if heartbeat is not None:
            touch_heartbeat(heartbeat, attempt=attempt, pid=None,
                            note=f"exited rc={rc}")
        log_line(f"supervisor: END attempt={attempt} rc={rc}")
        if rc == 0:
            return 0

        now = time.time()
        restarts.append(now)
        while restarts and now - restarts[0] > policy.restart_window_s:
            restarts.popleft()
        if len(restarts) > policy.max_restarts:
            if alert_path is not None:
                append_jsonl(alert_path, {
                    "schema": 1, "ts": now, "kind": "supervisor_crash_loop",
                    "cmd": list(cmd), "restarts": len(restarts),
                    "window_s": policy.restart_window_s, "last_rc": rc,
                })
            log_line(f"supervisor: ALERT crash-loop — {len(restarts)} "
                     f"restarts in {policy.restart_window_s:.0f}s; giving up")
            return rc
        delay = min(policy.backoff_max_s,
                    policy.backoff_init_s * (2 ** (len(restarts) - 1)))
        log_line(f"supervisor: RETRY attempt={attempt} rc={rc} "
                 f"restart#{len(restarts)} backoff={delay:.0f}s")
        _sleep_with_heartbeat(delay, heartbeat, policy.beat_interval_s,
                              note="backoff before restart")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="run a command under crash-restart supervision")
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--summary-log", default=None,
                        help="shared main log receiving summary lines only")
    parser.add_argument("--pause-file", default=None)
    parser.add_argument("--alert-path", default=None,
                        help="jsonl file for crash-loop alerts")
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--restart-window", type=float, default=600.0)
    parser.add_argument("--backoff-init", type=float, default=5.0)
    parser.add_argument("--backoff-max", type=float, default=300.0)
    parser.add_argument("--beat-interval", type=float, default=30.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="command after --")
    args = parser.parse_args(argv)

    cmd: List[str] = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command after --")

    policy = SupervisionPolicy(
        max_restarts=args.max_restarts,
        restart_window_s=args.restart_window,
        backoff_init_s=args.backoff_init,
        backoff_max_s=args.backoff_max,
        beat_interval_s=args.beat_interval,
    )
    return run_under_supervisor(
        cmd, heartbeat=Path(args.heartbeat), log_dir=Path(args.log_dir),
        policy=policy,
        pause_file=Path(args.pause_file) if args.pause_file else None,
        summary_log=Path(args.summary_log) if args.summary_log else None,
        alert_path=Path(args.alert_path) if args.alert_path else None,
    )


if __name__ == "__main__":
    sys.exit(main())
