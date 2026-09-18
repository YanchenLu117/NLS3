"""NLSS V8 cross-line infrastructure hardening library.

Reusable, stdlib-only building blocks implementing the infra SOP of
2026-09-01 as one shared package instead of per-line shell scripts:

  canary      endpoint health probe with rolling-window circuit breaker
  supervisor  restart-on-crash command wrapper (heartbeat + pause aware)
  watchdog    batch liveness / canary / rc-failure / disk monitor
  preflight   pre-launch checklist (registry hashes, endpoint smoke,
              disk guard, concurrency budget)
  loghygiene  per-job logs with a summary-only shared main log

Design invariants
-----------------
* **stdlib only** — runs in every project env and on both servers.
* **file-based coordination** — all cross-process state is small JSON
  files; any bash loop participates with a one-line existence test.
* **non-invasive** — deployable around already-running batches: the
  circuit breaker acts through a flag file that launchers choose to
  honor; nothing is injected into live processes.
* **ownership-aware flags** — a pause/flag file records its owner; the
  canary never removes a flag a human or the watchdog created.

The line-local prototypes in ``scripts/e4/`` (e4_canary.py,
e4_supervisor.sh, e4_heartbeat.sh) implement the same protocol; lines
are expected to migrate to this package at their next launch boundary.

Public names are imported lazily (PEP 562) so that
``python -m nlss.infra.<module>`` does not re-execute already-imported
submodules.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .canary import Canary, Probe, probe_chat, probe_models
    from .loghygiene import SummaryTee, rotate_by_size, tee_process_output
    from .preflight import Check, run_preflight_checks
    from .supervisor import (SupervisionPolicy, run_under_supervisor,
                             touch_heartbeat)
    from .watchdog import Watchdog

_LAZY = {
    "Canary": ("nlss.infra.canary", "Canary"),
    "Probe": ("nlss.infra.canary", "Probe"),
    "probe_chat": ("nlss.infra.canary", "probe_chat"),
    "probe_models": ("nlss.infra.canary", "probe_models"),
    "SupervisionPolicy": ("nlss.infra.supervisor", "SupervisionPolicy"),
    "run_under_supervisor": ("nlss.infra.supervisor", "run_under_supervisor"),
    "touch_heartbeat": ("nlss.infra.supervisor", "touch_heartbeat"),
    "Watchdog": ("nlss.infra.watchdog", "Watchdog"),
    "Check": ("nlss.infra.preflight", "Check"),
    "run_preflight_checks": ("nlss.infra.preflight", "run_preflight_checks"),
    "SummaryTee": ("nlss.infra.loghygiene", "SummaryTee"),
    "tee_process_output": ("nlss.infra.loghygiene", "tee_process_output"),
    "rotate_by_size": ("nlss.infra.loghygiene", "rotate_by_size"),
}


def __getattr__(name: str):
    import importlib

    if name in _LAZY:
        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals()) + list(_LAZY))
                  - {"TYPE_CHECKING"})
