# Archived campaigns (2026-08-22)

These single-file campaign baselines are SUPERSEDED by the unified comparator
registry in `src/nlss/baselines/` (A0-A4 + specialist block per
EXPERIMENT_PLAN_FINAL_V1 §5).  They are kept (not deleted) for reference /
recovery; the new run-driver and the baseline registry do NOT use them.  Do not
add new code here.

## Note (editor, 2026-08-22): commit 97ccee8 scope correction
The infra-archive commit 97ccee8 was created with `git add -A src/nlss/campaigns/`
and inadvertently ALSO swept in concurrent functional edits by a colleague to
`scienceagentbench/agent.py` and `runner.py` (propose_memory + arm-conditioned
generation + an API-key default change) that are NOT part of this archive intent.
Those edits were left IN PLACE (not reverted) — they are the colleagues workload