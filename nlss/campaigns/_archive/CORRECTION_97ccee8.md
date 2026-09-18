# Scope correction — commit 97ccee8 (editor, 2026-08-22)

The infra-archive commit 97ccee8 was created with `git add -A src/nlss/campaigns/`
and inadvertently ALSO swept in concurrent functional edits by a colleague to
`scienceagentbench/agent.py` and `runner.py` (propose_memory + arm-conditioned
generation + an API-key default change) that are NOT part of this archive intent.
Those edits were left IN PLACE (not reverted) — they are the colleague's workload
and may be intentional.

Going forward the infra/exec stream will use SCOPED `git add` (only the file(s)
being archived/edited), never `add -A` over a directory containing active
colleague work.  This document records the correction; the infra-archive itself
only intended the `_archive/` baseline moves.
