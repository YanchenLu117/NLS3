# nlss.infra — cross-line hardening library

Shared, stdlib-only implementation of the 2026-09-01 infra SOP.  One
package instead of per-line shell scripts; every module is runnable
(``PYTHONPATH=src python -m nlss.infra.<module> --help``) and importable.

| Module | Role | State it owns |
|---|---|---|
| `canary` | endpoint probe + rolling-window circuit breaker | `canary_state.json`, pause flag |
| `supervisor` | restart-on-crash wrapper (heartbeat, backoff, crash-loop guard, per-attempt logs, summary main log) | `heartbeat.json`, `attempt_*.log` |
| `watchdog` | heartbeat staleness / canary / new `rc!=0` / disk monitor | `watchdog_state.json`, `watchdog_alerts.jsonl`, its own pause flag |
| `preflight` | pre-launch checklist (registry sidecars, DECISION hashes, endpoint smoke, disk, concurrency budget) | `preflight_report.json` |
| `loghygiene` | per-job logs + summary-only main log + size rotation | — |

## Design invariants

1. **stdlib only** — identical behavior in `nlss-core`, `made-gpu`, and
   the vcc-4 base python, on c89 and vcc-4.
2. **file-based coordination** — cross-process state is small JSON; any
   bash loop joins the protocol with one line:
   `[ -f work/infra_shared/canary_pause.flag ] && { sleep 60; continue; }`
3. **non-invasive** — deployable around already-running batches; the
   breaker acts through flag files that launchers choose to honor.
4. **ownership-aware flags** — a flag records its `owner`; the canary
   never removes a flag a human or the watchdog created, and vice versa.

## Canonical deployment (per server)

```bash
mkdir -p work/infra_shared
chmod 600 work/infra_shared/.glm_api_key          # key file, never committed

# 1. canary (one per server; survives reboots via nohup)
PYTHONPATH=src nohup python -m nlss.infra.canary \
  --endpoint <YOUR-LLM-ENDPOINT> \
  --api-key-file work/infra_shared/.glm_api_key \
  --state work/infra_shared/canary_state.json \
  --pause-file work/infra_shared/canary_pause.flag \
  > work/infra_shared/canary_stdout.log 2>&1 &

# 2. run any batch under supervision (example: E-line batch)
PYTHONPATH=src python -m nlss.infra.supervisor \
  --heartbeat work/E_sciexplorer/runs/heartbeat.json \
  --log-dir work/E_sciexplorer/runs/supervisor \
  --summary-log work/E_sciexplorer/runs/batch_main.log \
  --pause-file work/infra_shared/canary_pause.flag \
  --alert-path work/infra_shared/infra_alerts.jsonl \
  -- bash work/E_sciexplorer/v8_scored_batch.sh 10 0

# 3. watchdog for that batch (separate terminal / nohup)
PYTHONPATH=src nohup python -m nlss.infra.watchdog \
  --heartbeat work/E_sciexplorer/runs/heartbeat.json \
  --canary-state work/infra_shared/canary_state.json \
  --rc-log work/E_sciexplorer/runs/batch_main.log \
  --disk /ssdwork=90 \
  --state work/E_sciexplorer/runs/watchdog_state.json \
  --alerts work/E_sciexplorer/runs/watchdog_alerts.jsonl \
  --pause-file work/infra_shared/watchdog_pause.flag \
  > work/E_sciexplorer/runs/watchdog_stdout.log 2>&1 &

# 4. preflight before every confirmatory launch (exit 0 = green)
PYTHONPATH=src python -m nlss.infra.preflight \
  --registry-dir work/A_hypospace/p0 \
  --decision-registry work/A_hypospace/p0/DECISION_REGISTRY_V3_20260831.json \
  --endpoint <YOUR-LLM-ENDPOINT> \
  --api-key-file work/infra_shared/.glm_api_key \
  --disk /usr/data=90 \
  --report work/A_hypospace/preflight_report.json
```

Worker loops that cannot run under the supervisor yet integrate with
the breaker by checking both pause flags before dispatching each job
(`canary_pause.flag` for endpoint trips, `watchdog_pause.flag` for
liveness/disk trips).

## Concurrency budget (daily check)

`work/infra_shared/budget.json`:

```json
{"ceiling": 20,
 "lines": {"E-line": {"pattern": "v8_scored_sciexplorer.py", "expect": 10},
           "A-line": {"pattern": "v8_run_hypospace.py", "expect": 6},
           "MADE":   {"pattern": "v8_scored_made.py", "expect": 2}}}
```

`python -m nlss.infra.preflight --budget work/infra_shared/budget.json`
(run on each server; the ceiling gate is the hard one, per-line drift is
a warning).

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests/infra -t .
```

## Migration from `scripts/e4/`

`scripts/e4/e4_canary.py`, `e4_supervisor.sh`, `e4_heartbeat.sh`
implement the same protocol line-locally (flag semantics are identical:
existence == open).  Migrate at the next launch boundary: stop the old
canary loop, start the shared one with the canonical paths above, and
launch future batches through `nlss.infra.supervisor`.  No running
batch needs to be restarted to adopt the watchdog or the flags.
