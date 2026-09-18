"""BH Morgan-fingerprint features (report 9.8 B3 / 9.12 BH-F).

The frozen strong-BO confrontation (GP-BO/DKL-BO/BALLET) needs a domain-native
feature representation on the *same BH landscape* as NLSS.  This provides the
component-Morgan representation used by the 2024 BH DKL study and quoted in the
technical report 9.8 B3:

    per component: Morgan fingerprint (radius 2, 512 bits)
    candidate     : concatenation over the 4 components -> 2048 dims

Features are outcome-blind (component identity + SMILES only; yields never
enter the representation), matching 4.4 information parity.
"""

from __future__ import annotations

import csv
import os
from typing import Mapping

import numpy as np
from rdkit import Chem
from rdkit import RDLogger as _RDLogger
_RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, DataStructs

from ..adapters.bh.data import _locate_data_file as _locate_bh_data

_COMPONENT_COLS = ("aryl_halide", "ligand", "base", "additive")
_FP_BITS = 512
_FP_RADIUS = 2


def _morgan_bitvec(smiles: str, n_bits: int = _FP_BITS, radius: int = _FP_RADIUS) -> np.ndarray:
    """Return a 512-bit Morgan bit-vector as a float32 0/1 array (outcome-blind)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _load_name_to_smiles(csv_path: str) -> dict:
    """Map each component name to its SMILES (from the measurement table)."""
    name2smiles: dict = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for name_col, smiles_col in (
                ("aryl_halide", "aryl_halide_smiles"),
                ("ligand", "ligand_smiles"),
                ("base", "base_smiles"),
                ("additive", "additive_smiles"),
            ):
                name = (row.get(name_col) or "").strip()
                smiles = (row.get(smiles_col) or "").strip()
                if name and name not in name2smiles:
                    name2smiles[name] = smiles
    return name2smiles


def morgan_features(oracle, data_path=None) -> dict:
    """Compute the 2048-d concatenated component-Morgan feature per candidate.

    ``data_path`` optionally overrides the CSV path; otherwise the oracle's own
    data adapter is used to locate the table.
    """
    path = None
    if data_path and os.path.isfile(data_path):
        path = data_path
    else:
        try:
            path = _locate_bh_data()
        except Exception:
            path = data_path
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "BH data_table.csv not found for Morgan features (path=%r); "
            "set NLSS_BH_DATA_TABLE or run from repo root." % (path,)
        )
    n2s = _load_name_to_smiles(path)
    out: dict = {}
    for cand, _ in oracle.data.records:
        parts = [_morgan_bitvec(n2s.get(cand[i], "")) for i in range(4)]
        out[cand] = np.concatenate(parts).astype(np.float32)
    return out


def morgan_similarity_regions(oracle, k: int = 5, sim_thresh: float | None = None):
    """Frozen, outcome-blind BH region geometry via Morgan k-NN (report 9.5).

    The one-factor substitution graph over the whole pool collapses to a single
    connected component, so RegionRecall degenerates (always 0/1).  Per report
    9.5 [PILOT-FREEZE], we break the over-connected geometry with a *scientific
    similarity filter*: connect candidates by Morgan-fingerprint k-nearest-
    neighbours (Tanimoto).  Regions = connected components of that k-NN graph.

    Outcome-blind: built from component SMILES/morphology only; yields never
    enter.  Shared across seeds/arms.
    """
    feats = morgan_features(oracle)
    pool = list(oracle.candidates)
    n = len(pool)
    X = np.asarray([feats[c] > 0.5 for c in pool], dtype=np.float32)  # 0/1 bits
    sums = X.sum(1)
    # k-NN by Tanimoto = |a & b| / |a | b|, vectorized in chunks
    adj = [set() for _ in range(n)]
    CH = 400
    for s0 in range(0, n, CH):
        block = X[s0:s0+CH]
        nb = block.shape[0]
        inter = (block @ X.T)                       # nb x n co-occurring bits
        denom = sums[None, :] + sums[s0:s0+CH][:, None] - inter
        sim = np.divide(inter, np.maximum(denom, 1e-9))
        np.fill_diagonal(sim[s0 - s0:], 0.0) if nb == 1 else None
        for r in range(nb):
            gi = s0 + r
            sim[r][gi] = 0.0
            order = np.argsort(-sim[r])[: k]
            for j in order:
                if sim[r][j] > 0:
                    adj[gi].add(int(j)); adj[int(j)].add(gi)
    # connected components via BFS
    comp = [-1] * n
    cid = 0
    unvisited = set(range(n))
    while unvisited:
        start = unvisited.pop()
        stack = [start]; comp[start] = cid
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in unvisited:
                    unvisited.discard(v); comp[v] = cid; stack.append(v)
        cid += 1
    return {c: comp[i] for i, c in enumerate(pool)}, cid
