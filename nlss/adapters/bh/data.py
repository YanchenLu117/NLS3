"""BH data loading — lazy, no-data-committed.

Loads the doylelab/rxnpredict ``data_table.csv`` from ``external/repos`` and
exposes the measured (candidate -> yield) records plus the component
vocabularies used for the outcome-blind one-hot features.

DATA-SOURCE BOOKKEEPING (口径标注):
    The implementation loads ``external/repos/bh_rxnpredict/data_table.csv``,
    which parses to **4,599 measured records** — i.e. the 16 aryl-halides x 4
    ligands x 3 bases x 24 additives grid (4,608 theoretical slots, 4,599 with
    measured yield after 9 missing/missing-yield rows are skipped).  All 4,599
    candidate tuples are unique.  This is the implementation-side oracle used
    by the adapter and pilot.
    The paper's technical-report §9.2 wording ("3,955 experimentally measured
    BH reaction records") is the **published doylelab count** and is retained
    verbatim as the frozen public-facing number; it is NOT the row count this
    implementation loads.  The difference (4,599 vs 3,955) is a published-vs-
    repo-subset discrepancy and must be stated honestly, never conflated.
"""

from __future__ import annotations

import csv
import os
from typing import Tuple

import numpy as np

# File ``data_table.csv`` columns (verified against the repo):
#   plate,row,col,base,base_cas_number,base_smiles,ligand,ligand_cas_number,
#   ligand_smiles,aryl_halide_number,aryl_halide,aryl_halide_smiles,
#   additive_number,additive,additive_smiles,product_smiles,yield
_YIELD_COL = "yield"
_COMPONENT_COLS = ("aryl_halide", "ligand", "base", "additive")
_EMPTY = ""

_DEFAULT_DATA_REL = "external/repos/bh_rxnpredict/data_table.csv"
_DEFAULT_DATA_ENV = "NLSS_BH_DATA_TABLE"

Candidate = Tuple[str, str, str, str]


def _locate_data_file() -> str | None:
    env = os.environ.get(_DEFAULT_DATA_ENV)
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), _DEFAULT_DATA_REL),
        os.path.join(here, "..", "..", "..", "..", "..", _DEFAULT_DATA_REL),
        os.path.join(here, "..", "..", "..", "..", "..", "..", _DEFAULT_DATA_REL),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return None


class BHData:
    """Lazily-loaded BH measured table: candidate -> yield, plus vocab.

    Only the measured yields are kept; gamma = q_0.95(y) is derived from them.
    """

    __slots__ = ("_records", "_loaded", "_n_loaded")

    def __init__(self) -> None:
        self._records: list[tuple[Candidate, float]] = []
        self._loaded = False

    def _ensure(self) -> None:
        if self._loaded:
            return
        path = _locate_data_file()
        if path is None:
            raise FileNotFoundError(
                f"BH data_table.csv not found; set {_DEFAULT_DATA_ENV} or run from the "
                f"NLSS_V7 repo root (expected at {_DEFAULT_DATA_REL})."
            )
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    y = float(row[_YIELD_COL])
                except (TypeError, ValueError):
                    continue  # skip NA / missing yield rows
                cand = tuple(
                    (str(row[c]).strip() if row[c] is not None else _EMPTY)
                    for c in _COMPONENT_COLS
                )
                self._records.append((cand, y))
        self._loaded = True

    @property
    def records(self) -> list[tuple[Candidate, float]]:
        self._ensure()
        return self._records

    @property
    def yields(self) -> np.ndarray:
        return self._yields()

    def _yields(self) -> np.ndarray:
        self._ensure()
        return np.asarray([y for _, y in self._records], dtype=np.float64)

    def gamma(self, q: float = 0.95) -> float:
        """Top-(1-q) yield threshold = q_0.95(y) by default (§9.4)."""
        ys = self._yields()
        if len(ys) == 0:
            raise ValueError("BH table empty; cannot compute gamma")
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0,1), got {q!r}")
        return float(np.quantile(ys, q))

    def effective_size(self) -> int:
        self._ensure()
        return len(self._records)

    def yield_of(self, cand: Candidate) -> float | None:
        """Return measured yield for ``cand`` or None if absent (reject, no reveal)."""
        self._ensure()
        for c, y in self._records:
            if c == cand:
                return y
        return None

    def population(self) -> set[Candidate]:
        self._ensure()
        return {c for c, _ in self._records}

    def contains(self, cand: Candidate) -> bool:
        self._ensure()
        return any(c == cand for c, _ in self._records)

    def component_vocab(self) -> dict[str, tuple[str, ...]]:
        """Deterministic per-component vocabularies for the one-hot features."""
        self._ensure()
        vocab: dict[str, list[str]] = {
            "aryl_halide": [],
            "ligand": [],
            "base": [],
            "additive": [],
        }
        for cand, _ in self._records:
            for i, key in enumerate(_COMPONENT_COLS):
                if cand[i] not in vocab[key]:
                    vocab[key].append(cand[i])
        return {k: tuple(sorted(v)) for k, v in vocab.items()}
