"""Per-job logs with a summary-only shared main log.

The single-mega-log pattern (one 951 MB scored_batch.log shared by
every worker) makes greps slow and gives no per-job isolation.  These
helpers let a batch write full output per job while the shared main log
receives only decision-relevant lines, and rotate oversized logs by
size.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Pattern, Tuple

DEFAULT_SUMMARY_RE = re.compile(
    r"\b(?:START|END|HEARTBEAT|ALERT|WARN|ERROR)\b|\brc=\d+")


class SummaryTee:
    """Classify log lines as summary vs. detail.

    >>> tee = SummaryTee(["[10:00] END job rc=0", "verbose noise"])
    >>> list(tee.classified())
    [('[10:00] END job rc=0\\n', True), ('verbose noise\\n', False)]
    """

    def __init__(self, lines: Iterable[str],
                 pattern: Pattern[str] = DEFAULT_SUMMARY_RE) -> None:
        self.lines = lines
        self.pattern = pattern

    def classified(self) -> Iterator[Tuple[str, bool]]:
        for line in self.lines:
            yield line, bool(self.pattern.search(line))


def tee_process_output(proc, job_log_path: Path,
                       main_log_path: Optional[Path] = None,
                       pattern: Pattern[str] = DEFAULT_SUMMARY_RE) -> None:
    """Drain ``proc.stdout`` into a per-job log (full) and a shared main
    log (summary lines only).  ``proc`` must have been started with
    ``stdout=PIPE, text=True``.
    """
    with open(job_log_path, "w") as job_log:
        main_log = open(main_log_path, "a") if main_log_path else None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                job_log.write(line)
                if main_log is not None and pattern.search(line):
                    main_log.write(line)
        finally:
            if main_log is not None:
                main_log.close()


def rotate_by_size(path: Path, max_bytes: int, keep: int = 5) -> bool:
    """Rotate ``path`` to ``path.1 .. path.<keep>`` when it exceeds
    ``max_bytes``; the oldest generation is dropped.  Returns True if a
    rotation happened.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size <= max_bytes:
        return False
    generations: List[Path] = [Path(f"{path}.{i}") for i in range(keep, 0, -1)]
    oldest = generations[0]
    if oldest.exists():
        oldest.unlink()
    for src, dst in zip(generations[1:], generations[:-1]):
        if src.exists():
            src.replace(dst)
    path.replace(generations[-1] if keep >= 1 else path)
    return True
