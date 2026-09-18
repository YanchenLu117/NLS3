"""GB1 data loading — §10.2 FROZEN measured-only boundary.

Primary source: the Wu 2016 eLife GB1 dataset as shipped in the CLADE clone at
``external/repos/gb1_clade/Input/GB1.xlsx`` (149,361 experimentally measured
variants).  Reading it requires ``openpyxl`` (not guaranteed in every env), so:

* If ``openpyxl`` is importable, the FULL measured set is loaded and a
  ``LoadedFrom`` metadata flag is set to ``"full_xlsx"``.
* Otherwise the adapter degrades to a small deterministic built-in demo subset
  (real measured values extracted from that file) so grounding/graph/oracle
  smoke tests still run.  This subset is NEVER presented as the full oracle;
  it is labeled ``"demo_subset"``.

Per §10.2, only the 149,361 measured variants are used; the 10,639 imputed
variants (eLife File 2) are NOT part of the oracle.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

DEFAULT_FULL_REL = "external/repos/gb1_clade/Input/GB1.xlsx"
DEFAULT_FULL_ENV = "NLSS_GB1_XLSX"

# Wild type of the 4-site GB1 combinatorial library (positions 39,40,41,54 of
# the 55-residue GB1 domain).  Verified from the first data row of GB1.xlsx.
WILD_TYPE = "VDGV"
N_SITES = 4

# Minimal real-measured demo subset (extracted from GB1.xlsx) used only when the
# full xlsx cannot be read.  Same schema as the full file: variant -> fitness.
DEMO_SUBSET: Dict[str, float] = {
    "VDGV": 1.0,
    "PKGL": 0.0173730887787,
    "DYPM": 0.0,
    "IPPA": 0.00147356631619,
    "PEWQ": 0.0,
    "TRYI": 0.0,
    "FWAA": 8.76196565571,
    "FYAA": 8.04515200416,
    "ANCA": 7.55686908836,
    "FWCA": 7.55466318627,
    "FWLG": 7.31265639682,
}

FitnessDict = Dict[str, float]


def _locate_xlsx() -> str | None:
    env = os.environ.get(DEFAULT_FULL_ENV)
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), DEFAULT_FULL_REL),
        os.path.join(os.getcwd(), "data", "gb1", "GB1.xlsx"),
        os.path.join(here, "..", "..", "..", "..", "..", DEFAULT_FULL_REL),
        os.path.join(here, "..", "..", "..", "..", "..", "..", DEFAULT_FULL_REL),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return None


def _load_full_xlsx(path: str) -> FitnessDict | None:
    """Read the 149,361 measured variants; returns None on any failure."""
    try:
        import openpyxl  # noqa: F401  (required to read xlsx)
        import pandas as pd
    except Exception:
        return None
    try:
        df = pd.read_excel(path, header=None, engine="openpyxl")
        # First row = header (Variants | HD | Count input | Count selected | Fitness)
        out: FitnessDict = {}
        for _, row in df.iloc[1:].iterrows():
            v = str(row[0]).strip()
            f = float(row[4])
            if v and len(v) == N_SITES:
                out[v] = f
        return out if out else None
    except Exception:
        return None


# Module-level lazy cache so repeated ``GB1Data()`` instances (e.g. across many
# pytest cases that each construct a fresh adapter) do NOT re-parse the 149k-row
# xlsx every time (~15s each).  The cache holds ``(fitness_dict, source)`` and is
# built once per process.  ``__slots__`` instances share it via this module-global.
_CACHE: tuple[FitnessDict | None, str] | None = None


class GB1Data:
    """Lazily-loaded GB1 measured fitness table."""

    __slots__ = ("_fitness", "_loaded", "_source")

    def __init__(self) -> None:
        global _CACHE
        if _CACHE is not None:
            self._fitness = dict(_CACHE[0]) if _CACHE[0] is not None else None
            self._source = _CACHE[1]
            self._loaded = True
            return
        self._fitness: FitnessDict | None = None
        self._loaded = False
        self._source = "unloaded"

    def _ensure(self) -> None:
        global _CACHE
        if self._loaded:
            return
        path = _locate_xlsx()
        full = _load_full_xlsx(path) if path else None
        if full is not None:
            self._fitness = full
            self._source = "full_xlsx"
        else:
            self._fitness = dict(DEMO_SUBSET)
            self._source = "demo_subset"
        _CACHE = (dict(self._fitness), self._source)
        self._loaded = True

    @property
    def fitness(self) -> FitnessDict:
        self._ensure()
        assert self._fitness is not None
        return self._fitness

    @property
    def source(self) -> str:
        self._ensure()
        return self._source

    def effective_size(self) -> int:
        self._ensure()
        return len(self._fitness or {})

    def is_full_measured(self) -> bool:
        self._ensure()
        return self._source == "full_xlsx"

    def fitness_of(self, variant: str) -> float | None:
        self._ensure()
        return (self._fitness or {}).get(variant)

    def population(self) -> set[str]:
        self._ensure()
        return set(self._fitness or {})

    def gamma(self, q: float = 0.95) -> float:
        """Fitness value threshold (top-``(1-q)``) passed as gamma (§10.2-ish)."""
        self._ensure()
        fs = np.asarray(list((self._fitness or {}).values()), dtype=np.float64)
        if len(fs) == 0:
            raise ValueError("GB1 table empty; cannot compute gamma")
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0,1), got {q!r}")
        return float(np.quantile(fs, q))
