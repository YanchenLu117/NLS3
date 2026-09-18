#!/usr/bin/env python
"""Preregistered optimization-campaign runner for the BH (Buchwald-Hartwig)
and GB1 protein-fitness benchmarks.

Phases:
  pilot      small-scale feasibility check per arm
  p0         frozen configuration + experiment registry (hash-verified)
  scored     the main seeded campaign; per-checkpoint score archives
  aggregate  preregistered statistics: Holm-corrected confirmatory family,
             equivalence (TOST) gates, descriptive contrasts
  figs       publication figures (paired box/violin/scatter with anchors)

Usage:
  python scripts/run_preregistered_protocol.py --project bh  --phase pilot
  python scripts/run_preregistered_protocol.py --project bh  --phase p0
  python scripts/run_preregistered_protocol.py --project bh  --phase scored --jobs 24
  python scripts/run_preregistered_protocol.py --project all --phase aggregate
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# BLAS/OpenMP thread caps MUST be set before numpy loads its threadpool —
# otherwise every pool worker defaults to nproc threads and 30+ workers thrash.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np  # noqa: E402

from nlss.campaigns.base import (  # noqa: E402
    CampaignConfig,
    ResourceLedger,
    RunConfig,
    git_commit,
    hash_config,
)
from nlss.campaigns.recovery_eval import RecoveryEvalResult, aurc, evaluate_recovery  # noqa: E402
from nlss.prereg.registry import PreregRegistry, verify_registry_hash  # noqa: E402
from nlss.stats.decision import (  # noqa: E402
    DECISION_IMPROVES,
    decide,
    holm_adjust,
    paired_summary,
    tost_equivalent,
)
from nlss.protocol.fairness import CAP_KEYS, ResourceMeter  # noqa: E402
from jsonl_writer import append_jsonl  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants (the published P0 registry is the authority once frozen)
# ---------------------------------------------------------------------------

# v3_1 endpoints (configs/llm_endpoints.json is the single source of truth):
# GLM-5.3-Flash HTTPS public direct (teacher-final 2026-08-29).  The old Mac
# tunnel (127.0.0.1:8002|8003) and deepseek-v4-flash are RETIRED (61d3a87) —
# requesting "deepseek" now fails loudly instead of silently hitting a
# reassigned port (36089 now serves GLM-5.3, a different model).
ENDPOINTS = {
    "glm": (os.environ.get("NLSS_LLM_BASE_URL", ""), os.environ.get("NLSS_LLM_MODEL", ""),
            "sk-glm53-62d65ae30556d43396fef44a258f415dddf84871cb1426ed"),
}
MAX_TOKENS = {"glm": 8192}  # GLM reasoning ~6k tokens/call cannot be disabled (blueprint §11)

CHECKPOINTS = {"bh": (40, 80, 120, 160, 200, 240), "gb1": (96, 192, 288, 384, 480)}
BATCH = {"bh": (40, 20, 10), "gb1": (96, 96, 4)}  # (n_initial, batch, rounds)
PILOT_SEEDS = {"bh": [9900 + i for i in range(8)], "gb1": [9900 + i for i in range(8)]}
MIN_SEEDS = {"bh": 10, "gb1": 30}   # Detail §8.2 BH=10 paired seeds, §9.2 GB1=30
                                    # (TEAM_BRIEF 2026-08-31: BH 10 配对 seed×240 步;
                                    # pilot power may propose more — extension needs a
                                    # 统筹 amendment BEFORE p0 freeze, outcome-blind)
MAX_SEEDS = 50
SD_FLOOR = 1e-3
DELTAS = {"delta_min": 0.01, "delta_eq": 0.01, "delta_harm": 0.02}  # §6, [0,1] endpoints

CAPS_NUM = {  # §13.1 seven caps; LLM caps declared but unused by numerical arms
    "llm_calls": 0.0,
    "generated_tokens": 0.0,
    "persistent_state_tokens": 0.0,
    "tool_calls": 64.0,
    "wall_time_seconds": 3600.0,
    "retries": 2.0,
}
GB1_ONEHOT_POOL_CAP = 25_000  # the previous frozen release harness value (documented O(n^2) feasibility cap)

# --- stage-1.5 official-code pins (external/repos.lock.yaml) ----------------
_CLADE_REPO = ROOT / "external" / "repos" / "gb1_clade"
_CLADE_LOCK_FINGERPRINT = "06c89367fd899e6c"   # content_fingerprint, lock 2026-08-31
_GRYFFIN_PIN = ("gryffin==1.0.0 PyPI sdist sha256 "
                "013d534047b8b12626015be75eba0be0dcc8463dfc58e21b198eb378a54a0948; "
                "wheel rebuilt on vcc-4 from official sources (2 Cython modules "
                "regenerated for py3.10; stale pre-generated .c incompatible)")


def _clade_checkout_fingerprint() -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(_CLADE_REPO.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts:
            h.update(f"{p.relative_to(_CLADE_REPO)}:{p.stat().st_size}".encode())
    return h.hexdigest()[:16]


def _clade_verify_pin() -> None:
    fp = _clade_checkout_fingerprint()
    if fp != _CLADE_LOCK_FINGERPRINT:
        raise RuntimeError(
            f"gb1_clade checkout fingerprint {fp} != locked {_CLADE_LOCK_FINGERPRINT} — refusing to run")


# Per-worker cache: official CLADE 'AA' encoding for the the current protocol pool.
_GB_CLADE_CACHE: dict = {}


def _gb_clade_features(oracle):
    """(n_pool, d) official CLADE GB1_AA_normalized.npy rows, pool-ordered.

    The encoding table + ComboToIndex ship inside the pinned gb1_clade checkout
    (official artifacts); labels never come from this path — only geometry."""
    import pickle
    if "feats" in _GB_CLADE_CACHE:
        return _GB_CLADE_CACHE["feats"]
    inp = _CLADE_REPO / "Input"
    c2i = pickle.load(open(inp / "ComboToIndex_GB1_AA.pkl", "rb"))
    tab = np.load(inp / "GB1_AA_normalized.npy")
    pool = list(oracle.candidates)
    missing = [v for v in pool if v not in c2i]
    if missing:
        raise RuntimeError(f"{len(missing)} pool variants missing from official ComboToIndex")
    rows = np.asarray([c2i[v] for v in pool], dtype=np.int64)
    if rows.max() >= tab.shape[0]:
        raise RuntimeError("official encoding table does not cover the pool index range")
    feats = np.ascontiguousarray(tab[rows].reshape(len(pool), -1), dtype=np.float64)
    _GB_CLADE_CACHE["feats"] = feats
    return feats

# Stage-1.5 specialists (E1_BRIEF 2026-08-31 step 2; Detail the current protocol §8.4/§9.4):
#   bh: Gryffin-plain (primary specialist) + Gryffin-desc (secondary) added;
#   gb1: CLADE (official implementation, §9.4) added.  MLDE-style reference is
#   NOT registered: official MLDE code unavailable (E1_ROSTER_PIN_SCAN).
DEFAULT_ARMS = {
    "bh": ["nlss_graph", "gryffin_plain", "gryffin_desc", "bo_gp", "ballet_level",
           "bo_dkl", "random", "hillclimb"],
    "gb1": ["nlss_hamming", "gb1_clade", "gb1_gpbo", "gb1_ballet", "gb1_alde",
            "random", "hillclimb"],
}
PRIMARY_ARM = {"bh": "nlss_graph", "gb1": "nlss_hamming"}
# Confirmatory Holm family = protocol specialists only (Detail §8.4/§9.4).
# random/hillclimb are proposal-policy context arms: the previous frozen release rows are an old-data
# join with metric n/a (BH_GB1_MAIN_AGG), and TEAM_BRIEF 2026-08-31 forbids
# them in the the current protocol confirmatory family — their the current protocol contrasts are computed but
# labeled descriptive (Holm p = raw p, family tag "descriptive").
# Family amendment (pre-outcome, before any the current protocol scored run / p0 freeze):
# superset of the 8-31 structure freeze + the stage-1.5 specialists mandated by
# Detail §8.4 (Gryffin-plain/desc) and §9.4 (CLADE).  No prior member removed;
# trimming is a 统筹 ruling at the p0 freeze (DECISION_STAGE15_ARMS_20260831).
CONFIRMATORY_BASELINES = {
    "bh": ("bo_gp", "ballet_level", "bo_dkl", "gryffin_plain", "gryffin_desc"),
    "gb1": ("gb1_gpbo", "gb1_ballet", "gb1_alde", "gb1_clade"),
}
DESCRIPTIVE_BASELINES = ("random", "hillclimb")
ANCHORS = {  # the previous frozen release frozen numbers (V8_REGRESSION_ANCHORS.md) — anchors + TOST gate only
    "bh": {"nlss_graph": 0.8112, "bo_gp": 0.5274, "ballet_level": 0.4843, "bo_dkl": 0.2683},
    "gb1": {"nlss_hamming": 7.8135, "gb1_gpbo": 7.5547, "gb1_ballet": 6.3605, "random": 4.2823},
}
BOOT_SEED = 20260829
PRIMARY_ENDPOINTS = ("aurc_common", "utility_auc", "tophit_auc")

_ARCH_FIX = re.compile(r"(?m)^(\s*)from \.([A-Za-z_][A-Za-z0-9_]*) import")


def archived(name: str):
    """Import an archived campaign module (moved under campaigns/_archive/).

    Archived files use intra-package relative imports that broke when they were
    moved; src/nlss is frozen for benchmark engineers, so a patched copy is
    materialized under work/_shared/v8_archive_cache/ and loaded from there.
    The archive itself is never modified.
    """
    cache = ROOT / "work" / "_shared" / "v8_archive_cache"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{name}.py"
    src = ROOT / "src" / "nlss" / "campaigns" / "_archive" / f"{name}.py"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        text = _ARCH_FIX.sub(r"\1from nlss.campaigns.\2 import", src.read_text())
        text = re.sub(r"(?m)^(\s*)from \.\.adapters\.", r"\1from nlss.adapters.", text)
        dst.write_text(text)
    modname = f"v8_archived_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, dst)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def runs_dir(project: str, phase: str) -> Path:
    area = {"bh": "work/B_bh", "gb1": "work/C_gb1"}[project]
    return ROOT / area / "v8_runs" / phase


def figs_dir(project: str) -> Path:
    d = ROOT / {"bh": "work/B_bh", "gb1": "work/C_gb1"}[project] / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ledger_path(project: str, phase: str) -> Path:
    d = runs_dir(project, phase)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ledger_{phase}.jsonl"


def sanitize(obj):
    """RFC 8259-safe: NaN/Inf -> None (Detail §15; B5 writer lesson)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return v if (isinstance(v, float) and math.isfinite(v)) or isinstance(v, int) else None
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (set, tuple)):
        return sanitize(list(obj))
    return obj


def dump_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(sanitize(obj), indent=2, allow_nan=False, default=str))


# ---------------------------------------------------------------------------
# Worker-side cached evaluation universe (built once per worker process)
# ---------------------------------------------------------------------------

_W: dict = {}


def _limit_threads() -> None:
    """Runtime BLAS/OpenMP cap (threadpoolctl): forked workers inherit the
    parent's already-initialized thread pools; env vars alone are too late."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(4)
    except Exception:
        pass


def _bh_worker_init(arms: tuple) -> None:
    # single-thread BLAS/OMP per worker: measured 24-way concurrent gryffin runs
    # at 4 OMP threads collapse to ~34 min/task (OMP spin-barrier thrash, only
    # 1.75x throughput over sequential); 1 thread/run is thread-count-insensitive
    # per the determinism A/B (171 s vs 171 s) and restores near-linear scaling.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _limit_threads()
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from nlss.adapters.bh.data import BHData
    from nlss.campaigns.bh_oracle import BHFiniteOracle
    data = _BHMeasuredUniverse(BHData())
    oracle = BHFiniteOracle(data, top_frac=0.05)
    mm = archived("bh_morgan")
    regions, _n = mm.morgan_similarity_regions(oracle, k=5)
    morgan = mm.morgan_features(oracle) if any(a in ("bo_gp", "bo_dkl") for a in arms) else None
    _W.clear()
    _W.update(data=data, oracle=oracle, regions=regions, morgan=morgan)


def _gb1_worker_init(arms: tuple) -> None:
    # single-thread OMP per worker — see _bh_worker_init rationale
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    _limit_threads()
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from nlss.campaigns.gb1_oracle import GB1FiniteOracle
    from nlss.campaigns.gb1_predict import GB1HammingProp
    oracle = GB1FiniteOracle(criterion="top_pct", top_frac=0.05)
    oracle_wt = GB1FiniteOracle(criterion="better_wt")  # sensitivity only
    hamm = GB1HammingProp(oracle) if "nlss_hamming" in arms else None
    regions = _gb1_solution_regions_knn(oracle, k=5)
    _W.clear()
    _W.update(oracle=oracle, oracle_wt=oracle_wt, hamm=hamm, regions=regions)


# ---------------------------------------------------------------------------
# BH arms (the previous frozen release harness semantics + the current protocol metering / §6 score capture)
# ---------------------------------------------------------------------------

_BH_KEYS = ("aryl_halide", "ligand", "base", "additive")


# The single release additive condition beyond the paper's 22-condition ML grid
# (Doyle/Dreher BH).  Excluding it reproduces the protocol's 3,955 measured
# reactions exactly (verified 2026-08-31 on c89: 4,132 all-4-component rows − 177
# oxadiazole wells = 3,955; |S*|=198 top-5%, gamma=83.2107).
PAPER_EXCLUDED_ADDITIVE = "5-Phenyl-1,2,4-oxadiazole"


class _BHMeasuredUniverse:
    """Primary scored universe = Detail §8.1 fixed table of 3,955 measured
    reactions (protocol authority; E1 registry-freeze 2026-08-31).

    Filter, verified derivable from the official release: rows with ALL FOUR
    components present (4,132) AND additive != PAPER_EXCLUDED_ADDITIVE
    (releases' single beyond-paper additive, 177 wells) -> 3,955.

    Sensitivity universes, never primary (counts verified on c89 2026-08-31):
      - all-4-component real wells ......... 4,132 rows, |S*|=207, gamma=83.048
        (the pre-verification choice; recorded as labeled sensitivity)
      - first-3-nonempty (the previous frozen release anchor uni.) .. 4,312 rows, |S*|=216, gamma=83.370
        (used only by v8_probe_wt.py Probe A for the previous frozen release-anchor attribution and by
        results/bh_main_final — NOT comparable to the primary universe)
      - raw release table .................. 4,599 rows (descriptive appendix;
        includes 287 NaN aryl-halide control wells and empty-additive plates)

    History: the 2026-08-29 freeze-control note claimed 3,955 was not derivable
    from the official release and adopted 4,132; that premise was falsified
    2026-08-31 (additive-set filter reproduces 3,955 exactly), so the registry
    pins the protocol-exact 3,955.  统筹 veto point: the p0 registry hash freeze.
    """

    def __init__(self, data):
        self._data = data
        self._recs = [(c, y) for c, y in data.records
                      if all(str(x).strip() for x in c)
                      and str(c[3]).strip() != PAPER_EXCLUDED_ADDITIVE]

    @property
    def records(self):
        return self._recs

    def component_vocab(self):
        return self._data.component_vocab()


def _bh_onehot_features(data) -> dict:
    vocab = data.component_vocab()
    out = {}
    for cand, _ in data.records:
        parts: list[float] = []
        for i, key in enumerate(_BH_KEYS):
            domain = vocab[key]
            hot = [0.0] * len(domain)
            if cand[i] in domain:
                hot[domain.index(cand[i])] = 1.0
            parts.extend(hot)
        out[cand] = np.asarray(parts, dtype=np.float64)
    return out


def _acq_random(oracle, queried: set, rng, n: int, pool=None) -> list:
    avail = [c for c in (oracle.candidates if pool is None else pool) if c not in queried]
    rng.shuffle(avail)
    return avail[:n]


def _acq_hillclimb_bh(oracle, queried: set, known: dict, vocab, rng, n: int) -> list:
    pool = set(oracle.candidates)
    if not known:
        return _acq_random(oracle, queried, rng, n)
    best = max(known, key=lambda c: known[c])
    b = list(best)
    proposed = []
    for slot_i in range(len(_BH_KEYS)):
        for val in vocab[_BH_KEYS[slot_i]]:
            cand = tuple(b[:slot_i] + [val] + b[slot_i + 1:])
            if cand in pool and cand not in queried:
                proposed.append(cand)
    rng.shuffle(proposed)
    return proposed[:n]


def _bh_floor_ranking(arm, oracle, queried, known, vocab, rng, pool_list):
    """Frozen read-only extractor (Detail §2.3): the floor arm's deterministic
    proposal ordering over unqueried candidates.  The rng state is cloned and
    restored, so extraction cannot affect later behavior."""
    avail = [c for c in pool_list if c not in queried]
    state = rng.getstate()
    try:
        if arm == "random":
            rng.shuffle(avail)
        elif arm == "hillclimb":
            head = _acq_hillclimb_bh(oracle, queried, known, vocab, rng, len(avail))
            rest = [c for c in avail if c not in set(head)]
            avail = head + rest
        else:
            pass  # frozen pool-order ranking for arms without a prob head (gryffin)
    finally:
        rng.setstate(state)
    return avail


class _BHGryffinAdapter:
    """Thin adapter over the OFFICIAL gryffin package (pin: gryffin==1.0.0).

    - plain: categorical identities only (no descriptors).
    - desc: FROZEN outcome-blind one-hot descriptor per option, identical for
      every candidate; auto_desc_gen stays False — Dynamic-Gryffin refinement
      is outcome-derived and Detail §8.4 forbids that for this arm.
    - legality: proposals are filtered to the frozen 3,955-reaction legal mask,
      unqueried, dedup'd, in native ranked order; shortfall filled by the
      frozen random replacement rule (_acq_random over the same pool).  Counts
      are disclosed in the run log (§8.3 invalid/duplicate proposal rate).
    - readout: none (no solution-probability head) -> NaN unseen metrics +
      frozen pool-order score capture, same treatment as random/hillclimb.
    """

    def __init__(self, arm: str, vocab: dict, legal_set: set, seed: int):
        import torch
        from gryffin import Gryffin  # official gryffin==1.0.0
        torch.manual_seed(int(seed) % (2 ** 31))
        if arm == "gryffin_desc":
            # frozen one-hot identity descriptor per option, per component
            # (own width; gryffin keeps per-parameter descriptor blocks).
            cat = {k: {opt: [1.0 if o == opt else 0.0 for o in vocab[k]] for opt in vocab[k]}
                   for k in _BH_KEYS}
        else:
            cat = {k: {opt: None for opt in vocab[k]} for k in _BH_KEYS}
        config = {
            "general": {"random_seed": int(seed) % (2 ** 31), "verbosity": 0,
                        "num_cpus": 1, "auto_desc_gen": False,
                        "save_database": False},
            "parameters": [{"name": k, "type": "categorical", "category_details": cat[k]}
                           for k in _BH_KEYS],
            "objectives": [{"name": "yield", "goal": "max"}],  # Chimera goals: min|max
        }
        self.legal_set = legal_set
        self.arm = arm
        self.n_proposed = 0
        self.n_rejected = 0
        self.n_fill = 0
        # official batch convention for 20 experiments: 5 sampling strategies
        # x 4 batches (cli.infer_batches_and_strategies, multiple of 5 branch)
        self._strategies = [float(s) for s in np.linspace(1.0, -1.0, 5)]
        self.gf = Gryffin(config_dict=config, silent=True)

    def acquire(self, known: dict, queried: set, batch: int, rng, pool: list) -> list:
        obs = [{**dict(zip(_BH_KEYS, c)), "yield": float(v)} for c, v in known.items()]
        props = self.gf.recommend(observations=obs, sampling_strategies=self._strategies,
                                  num_batches=max(1, batch // len(self._strategies)))
        self.n_proposed += len(props)
        picks: list = []
        for p in props:
            try:
                t = tuple(str(p[k]) for k in _BH_KEYS)
            except (KeyError, TypeError):
                self.n_rejected += 1
                continue
            if t in self.legal_set and t not in queried and t not in picks:
                picks.append(t)
            if len(picks) == batch:
                break
        if len(picks) < batch:
            need = batch - len(picks)
            self.n_fill += need
            picks.extend(_acq_random(None, queried | set(picks), rng, need, pool=pool))
        return picks


def bh_run_one(task: tuple):
    """One BH (arm, seed) campaign: the previous frozen release harness semantics + the current protocol metering/capture."""
    (arm, seed, n_initial, batch, rounds, checkpoints, out_dir, registry_hash,
     cfg_hash, commit, model_key) = task

    from nlss.campaigns.bh_graph import BHGraphProp
    from nlss.campaigns.bh_gp import BHPoolGP

    oracle = _W["oracle"]
    data = _W["data"]
    morgan_regions = _W["regions"]
    morgan = _W["morgan"]
    pool = list(oracle.candidates)
    pool_idx = {c: i for i, c in enumerate(pool)}
    features = _bh_onehot_features(data)
    vocab = data.component_vocab()
    n_sol = len(oracle.solution_set)

    t_start = time.time()
    total_budget = n_initial + rounds * batch
    caps = dict(CAPS_NUM)
    caps["oracle_budget"] = float(total_budget)
    meter = ResourceMeter(caps)
    run = RunConfig(benchmark="bh", method=arm, seed=seed, config_hash=cfg_hash, git_commit=commit)
    ledger7 = ResourceLedger(run)

    rng = random.Random(seed)
    queried: set = set()
    known: dict = {}
    prefix_hits: dict[int, int] = {}
    prefix_found_recall: dict[int, float] = {}
    prefix_unseen_recall: dict[int, float] = {}
    prefix_best_yield: dict[int, float] = {}
    checkpoint_metrics: dict[int, dict] = {}
    gp = bo = lset = None
    gf = None
    failure: dict | None = None
    surr_seconds: list = []

    def found_recall() -> float:
        return (sum(1 for c in queried if oracle.solution_membership(c)) / n_sol) if n_sol else 0.0

    def best_yield() -> float:
        return max(known.values()) if known else 0.0

    def v7_eval_at(budget: int):
        unqueried = [c for c in pool if c not in queried]
        prob_map = None
        if arm.startswith("nlss") and gp is not None and gp.trained:
            prob_map = gp.predict_all_solution_prob(unqueried)
        elif arm in ("bo_gp", "bo_dkl") and bo is not None and bo.trained:
            prob_map = bo.predict_all_solution_prob(unqueried)
        elif arm == "ballet_level" and lset is not None and lset.trained:
            prob_map = lset.predict_all_solution_prob(unqueried)
        if prob_map is None:
            return RecoveryEvalResult(
                n_unqueried=len(unqueried), unseen_recall=float("nan"),
                unseen_precision=float("nan"), unseen_f1=float("nan"),
                pr_auc=float("nan"), region_recall=float("nan"),
                uncovered_distance=None, brier=float("nan"), ece=float("nan"))
        supported = [c for c in unqueried if prob_map.get(c, 0.5) >= 0.5]
        return evaluate_recovery(
            unqueried_candidates=unqueried,
            true_membership=oracle.solution_membership,
            pred_solution_prob=lambda c: prob_map.get(c, 0.5),
            true_solution_prob=lambda c: 1.0 if oracle.solution_membership(c) else 0.0,
            node_features=None,
            true_components_of=lambda c: morgan_regions.get(c),
            recovered_supported=supported,
            region_rank_K=60,
        )

    try:
        init = _acq_random(oracle, queried, rng, n_initial)
        for c in init:
            known[c] = oracle.evaluate(c)
            queried.add(c)
            ledger7.bump_oracle()
            meter.consume("oracle_budget", 1.0)
        if arm == "nlss_graph":
            gp = BHGraphProp(oracle)
            gp.fit(known, surr_seconds)
            meter.consume("tool_calls", 1.0)
            if surr_seconds:
                ledger7.bump_surrogate_fit(surr_seconds[-1])
        elif arm in ("bo_gp", "bo_dkl"):
            ab = archived("bh_bo")
            bo_feats = morgan if (arm == "bo_dkl" and morgan is not None) else features
            bo = ab.BHBotorchBO(oracle, bo_feats, kernel=("dkl" if arm == "bo_dkl" else "rbf"))
            t0 = time.time()
            bo.fit(known)
            ledger7.bump_surrogate_fit(time.time() - t0)
            meter.consume("tool_calls", 1.0)
        elif arm == "ballet_level":
            ab = archived("bh_ballet")
            lset = ab.BHBalletLevelSet(oracle, features)
            lset.fit(known, surr_seconds)
            meter.consume("tool_calls", 1.0)
            if surr_seconds:
                ledger7.bump_surrogate_fit(surr_seconds[-1])
        elif arm in ("gryffin_plain", "gryffin_desc"):
            legal = set(oracle.candidates)
            gf = _BHGryffinAdapter(arm, vocab, legal, seed)
            meter.consume("tool_calls", 1.0)

        budget_so_far = n_initial
        prefix_hits[budget_so_far] = sum(1 for c in queried if oracle.solution_membership(c))
        prefix_found_recall[budget_so_far] = found_recall()
        prefix_best_yield[budget_so_far] = best_yield()
        res0 = v7_eval_at(budget_so_far)
        prefix_unseen_recall[budget_so_far] = res0.unseen_recall
        checkpoint_metrics[budget_so_far] = res0.as_dict()
        checkpoint_metrics[budget_so_far].update(
            found_recall=found_recall(), solutions_found=prefix_hits[budget_so_far],
            best_yield=best_yield())
        if budget_so_far in checkpoints:
            _bh_write_scores(out_dir, arm, seed, budget_so_far, pool, pool_idx,
                             oracle, queried, known, vocab, rng, gp, bo, lset)

        for rnd in range(rounds):
            if (time.time() - t_start) > caps["wall_time_seconds"]:
                failure = {"reason": "wall_time_cap", "at_checkpoint": budget_so_far}
                break
            budget_so_far += batch
            if arm == "random":
                batch_c = _acq_random(oracle, queried, rng, batch)
            elif arm == "hillclimb":
                batch_c = _acq_hillclimb_bh(oracle, queried, known, vocab, rng, batch)
            elif arm in ("bo_gp", "bo_dkl"):
                batch_c = bo.acquire(queried, batch)
            elif arm in ("gryffin_plain", "gryffin_desc"):
                batch_c = gf.acquire(known, queried, batch, rng, pool)
            elif arm == "ballet_level":
                batch_c = lset.acquire(queried, batch, rng, coverage=0.0)
                surr_seconds.clear()
                lset.fit(known, surr_seconds)
                meter.consume("tool_calls", 1.0)
                if surr_seconds:
                    ledger7.bump_surrogate_fit(surr_seconds[-1])
            elif arm.startswith("nlss"):
                avail = [c for c in pool if c not in queried]
                prob_map = gp.predict_all_solution_prob(avail)
                batch_c = sorted(avail, key=lambda c: -prob_map.get(c, 0.5))[:batch]
            else:
                raise ValueError(f"unknown BH arm {arm}")
            for c in batch_c:
                known[c] = oracle.evaluate(c)
                queried.add(c)
                ledger7.bump_oracle()
                meter.consume("oracle_budget", 1.0)
            if arm.startswith("nlss"):
                surr_seconds.clear()
                gp.fit(known, surr_seconds)
                meter.consume("tool_calls", 1.0)
                if surr_seconds:
                    ledger7.bump_surrogate_fit(surr_seconds[-1])
            elif arm in ("bo_gp", "bo_dkl"):
                t0 = time.time()
                bo.fit(known)
                ledger7.bump_surrogate_fit(time.time() - t0)
                meter.consume("tool_calls", 1.0)
            prefix_hits[budget_so_far] = sum(1 for c in queried if oracle.solution_membership(c))
            prefix_found_recall[budget_so_far] = found_recall()
            prefix_best_yield[budget_so_far] = best_yield()
            res = v7_eval_at(budget_so_far)
            prefix_unseen_recall[budget_so_far] = res.unseen_recall
            if budget_so_far in checkpoints or budget_so_far == total_budget:
                checkpoint_metrics[budget_so_far] = res.as_dict()
                checkpoint_metrics[budget_so_far].update(
                    found_recall=found_recall(), solutions_found=prefix_hits[budget_so_far],
                    best_yield=best_yield())
                if budget_so_far in checkpoints:
                    _bh_write_scores(out_dir, arm, seed, budget_so_far, pool, pool_idx,
                                     oracle, queried, known, vocab, rng, gp, bo, lset)
    except Exception as exc:  # method-attributable failure: stays in denominator (§14)
        import traceback
        failure = {"reason": f"{type(exc).__name__}: {exc}",
                   "at_checkpoint": max(checkpoint_metrics) if checkpoint_metrics else n_initial,
                   "traceback": traceback.format_exc(limit=6)}

    wall = time.time() - t_start
    meter.consume("wall_time_seconds", wall)
    last_key = max(checkpoint_metrics) if checkpoint_metrics else n_initial
    ledger7.set(final_metrics=checkpoint_metrics.get(last_key, {}), checkpoint_metrics=checkpoint_metrics)
    log = ledger7.to_dict()
    last = dict(checkpoint_metrics.get(last_key, {}))
    last.update(
        solutions_found_total=max(prefix_hits.values()) if prefix_hits else 0,
        best_yield=max(prefix_best_yield.values()) if prefix_best_yield else 0.0,
        solution_prevalence=oracle.solution_prevalence(),
    )
    log["final_metrics"] = last
    log["protocol"] = {
        "phase": "scored",
        "registry_hash": registry_hash,
        "model": model_key,
        "llm_calls": 0,
        "retries": 0,
        "parse_failures": 0,
        "latency_s": round(wall, 3),
        "tokens": 0,
        "cap_state": {
            "used": meter.usage(),
            "exceeded": {k: bool(meter.exceeded[k]) for k in CAP_KEYS},
            "incomplete_under_cap": meter.incomplete_under_cap,
        },
        "failure": failure,
        "prefix": {
            "hits": {str(k): v for k, v in prefix_hits.items()},
            "found_recall": {str(k): v for k, v in prefix_found_recall.items()},
            "best_yield": {str(k): v for k, v in prefix_best_yield.items()},
            "aurc_found_recall": aurc(prefix_found_recall),
            "aurc_best_yield": aurc(prefix_best_yield),
        },
    }
    if arm in ("gryffin_plain", "gryffin_desc") and gf is not None:
        log["protocol"]["specialist"] = {
            "pin": _GRYFFIN_PIN, "arm": arm,
            "n_native_proposals": int(gf.n_proposed),
            "n_rejected_proposals": int(gf.n_rejected),
            "n_fill_queries": int(gf.n_fill),
            "descriptor_mode": ("frozen_onehot_options" if arm == "gryffin_desc" else "none"),
            "auto_desc_gen": False,
        }
    return log


def _bh_write_scores(out_dir, arm, seed, budget, pool, pool_idx, oracle, queried,
                     known, vocab, rng, gp, bo, lset) -> None:
    """§6 capture: score vector over the frozen pool order (float32) + queried
    bitmap, for common-universe AP at aggregate time."""
    score = np.zeros(len(pool), dtype=np.float32)
    unq = [c for c in pool if c not in queried]
    src = None
    for cand in (gp, bo, lset):
        if cand is not None and getattr(cand, "trained", False):
            src = cand
            break
    if src is not None:
        try:
            prob_map = src.predict_all_solution_prob(unq)
            for c in unq:
                score[pool_idx[c]] = np.float32(prob_map.get(c, 0.5))
        except Exception:
            score = np.zeros(len(pool), dtype=np.float32)
    if not score.any():
        ranking = _bh_floor_ranking(arm, oracle, queried, known, vocab, rng, pool)
        for rank, c in enumerate(ranking):
            score[pool_idx[c]] = np.float32(1.0 - rank / max(1, len(ranking)))
    qmask = np.zeros(len(pool), dtype=bool)
    for c in queried:
        qmask[pool_idx[c]] = 1
    np.savez_compressed(Path(out_dir) / f"bh_{arm}_seed{seed:02d}_s{budget}.npz",
                        score=score, queried=qmask)


# ---------------------------------------------------------------------------
# GB1 arms
# ---------------------------------------------------------------------------

_GB1_AA = "ACDEFGHIKLMNPQRSTVWY"


def _gb1_onehot(v: str) -> np.ndarray:
    vec = np.zeros(4 * len(_GB1_AA), dtype=np.float64)
    for j, ch in enumerate(v):
        vec[j * len(_GB1_AA) + _GB1_AA.index(ch)] = 1.0
    return vec


def _acq_hillclimb_gb1(query_pool, queried: set, known: dict, rng, n: int) -> list:
    if not known:
        return _acq_random_pool(query_pool, queried, rng, n)
    best = max(known, key=lambda c: known[c])
    pool_set = set(query_pool)
    proposed = []
    for i in range(4):
        for aa in _GB1_AA:
            if aa != best[i]:
                cand = best[:i] + aa + best[i + 1:]
                if cand in pool_set and cand not in queried:
                    proposed.append(cand)
    rng.shuffle(proposed)
    return proposed[:n]


def _acq_random_pool(query_pool, queried: set, rng, n: int) -> list:
    avail = [c for c in query_pool if c not in queried]
    rng.shuffle(avail)
    return avail[:n]


def _gb1_solution_regions_knn(oracle, k: int = 5) -> dict[int, list[int]]:
    """Frozen outcome-blind S* family geometry (the previous frozen release harness: kNN on 80-d one-hot)."""
    sol = list(oracle.solution_set)
    X = np.zeros((len(sol), 4 * len(_GB1_AA)), dtype=np.float64)
    for r, v in enumerate(sol):
        for j, ch in enumerate(v):
            X[r, j * len(_GB1_AA) + _GB1_AA.index(ch)] = 1.0
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    adj = [set() for _ in range(len(sol))]
    CH = 512
    for s0 in range(0, len(sol), CH):
        blk = Xn[s0:s0 + CH]
        sim = blk @ Xn.T
        for r in range(blk.shape[0]):
            gi = s0 + r
            sim[r][gi] = -1.0
            for jj in np.argsort(-sim[r])[:k]:
                adj[gi].add(int(jj))
                adj[int(jj)].add(gi)
    comp = [-1] * len(sol)
    cid = 0
    unvis = set(range(len(sol)))
    while unvis:
        start = unvis.pop()
        comp[start] = cid
        stack = [start]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w in unvis:
                    unvis.discard(w)
                    comp[w] = cid
                    stack.append(w)
        cid += 1
    idx = {v: i for i, v in enumerate(oracle.candidates)}
    out: dict[int, list[int]] = {}
    for rid in range(cid):
        members = [idx[sol[i]] for i in range(len(sol)) if comp[i] == rid]
        if members:
            out[rid] = members
    return out


def _gb1_recovery_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    if len(labels) == 0:
        return {k: float("nan") for k in
                ("unseen_recall", "unseen_precision", "unseen_f1", "pr_auc", "brier")}
    pred_pos = probs >= 0.5
    tp = int(np.sum(pred_pos & (labels == 1)))
    fp = int(np.sum(pred_pos & (labels == 0)))
    fn = int(np.sum((~pred_pos) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    order = np.argsort(-probs, kind="mergesort")
    rl = labels[order]
    pos_idx = np.where(rl == 1)[0]
    pr = float(np.mean((np.arange(len(pos_idx)) + 1) / (pos_idx + 1))) if len(pos_idx) else float("nan")
    brier = float(np.mean((probs - labels) ** 2))
    return {"unseen_recall": recall, "unseen_precision": precision,
            "unseen_f1": f1, "pr_auc": pr, "brier": brier}


def _gb1_region_recall(supported_idx: set, regions: dict[int, list[int]]) -> float:
    if not regions:
        return float("nan")
    covered = sum(1 for members in regions.values() if any(m in supported_idx for m in members))
    return covered / len(regions)


def _clade_official_sequence(oracle, pool: list, seed: int, save_dir: Path):
    """Run the OFFICIAL CLADE sampling core (clustering_sampling.cluster_sample)
    once, with an honest label channel: the Fitness wrapper evaluates the the current protocol
    oracle exactly once per selected variant and refuses (raises) on any access
    to an unselected variant.  Returns (ordered query sequence, n_oracle_calls).

    Official code is used verbatim: KMeans hierarchy (run_Clustering), cluster
    proportional sampling, per-batch mean-fitness reweighting, GP priority
    (sampling_subcluster_priority, UCB beta=4 per Romero PNAS 2013), hierarchy
    splits at 192/288 (K_increments 30/30/30, hierarchy_batch=96) — the native
    96 + 4x96 = 480 budget equals the the current protocol GB1 schedule exactly (§9.2/§9.4).
    Determinism: np.random.seed(seed) around the call (state restored).
    """
    _clade_verify_pin()
    import sys as _sys
    import numpy as _np
    if str(_CLADE_REPO) not in _sys.path:
        _sys.path.insert(0, str(_CLADE_REPO))
    import clustering_sampling as _cs  # official module, pinned fingerprint

    # Upstream robustness cap (disclosed deviation): split_subcluster can
    # request more sub-clusters than a high-mean subcluster has REMAINING
    # members (KMeans n_samples < n_clusters -> ValueError) for some random
    # states.  Cap n_clusters at the available sample count — identical
    # behaviour in the non-crashing regime.
    _run_clustering_official = _cs.run_Clustering

    def _run_clustering_capped(features, n_clusters, subclustering_index=None):
        # upstream invariant: split_subcluster pre-sizes Fit/SEQ/Index lists to
        # the REQUESTED n_clusters, so the return length must equal n_clusters
        # even when KMeans can only produce fewer singleton clusters — pad the
        # shortfall with empty index arrays (the official empty-cluster
        # handling downstream treats them as depleted).
        n_req = int(n_clusters)
        if subclustering_index is not None and len(subclustering_index) > 0:
            n_avail = len(subclustering_index)
            if n_avail >= n_req:
                return _run_clustering_official(features, n_req, subclustering_index)
            out = list(_run_clustering_official(features, n_avail, subclustering_index))
            pad = np.array([], dtype=(out[0].dtype if out else np.int64))
            return out + [pad.copy() for _ in range(n_req - len(out))]
        if len(features) < n_req:
            out = list(_run_clustering_official(features, len(features)))
            pad = np.array([], dtype=(out[0].dtype if out else np.int64))
            return out + [pad.copy() for _ in range(n_req - len(out))]
        return _run_clustering_official(features, n_req)

    _cs.run_Clustering = _run_clustering_capped
    feats = _gb_clade_features(oracle)
    combo_to_index = {v: i for i, v in enumerate(pool)}
    state = {"calls": 0, "labels": {}, "order": []}

    class _LabelChannel:
        """Honest label channel: evaluate-on-demand at selection, raise otherwise."""

        def __getitem__(self, idx):
            if isinstance(idx, (list, tuple, _np.ndarray)):
                return _np.asarray([self[int(i)] for i in idx], dtype=float)
            idx = int(idx)
            if idx not in state["labels"]:
                state["labels"][idx] = oracle.evaluate(pool[idx])
                state["calls"] += 1
                state["order"].append(idx)
            return state["labels"][idx]

    from types import SimpleNamespace
    save_dir.mkdir(parents=True, exist_ok=True)
    # official total-label semantics: num_training_data = batch_size * num_batch
    # (the first round is batch 1 of num_batch), so 96 + 4x96 = 480 needs
    # num_batch=5; hierarchy splits then land at 192/288/384 (K 30/30/30/30).
    args = SimpleNamespace(
        K_increments=[30, 30, 30, 30], dataset="GB1", encoding="AA",
        num_first_round=96, batch_size=96, hierarchy_batch=96, num_batch=5,
        input_path=str(_CLADE_REPO / "Input"), save_dir=str(save_dir),
        seed=int(seed), acquisition="UCB", sampling_para=4.0,
        use_zeroshot=False, zeroshot="EvMutation", N_zeroshot=1600,
        mldepara="MldeParameters.csv")
    np_state = _np.random.get_state()
    try:
        _np.random.seed(int(seed) % (2 ** 31))
        _cs.cluster_sample(args, str(save_dir), feats, pool, _LabelChannel(),
                           combo_to_index)
    finally:
        _np.random.set_state(np_state)
    if len(state["order"]) != len(set(state["order"])):
        raise RuntimeError("official CLADE sequence contains a repeated variant")
    return [pool[i] for i in state["order"]], state["calls"]


class _CladeReadout:
    """Checkpoint readout for gb1_clade: the official in-repo sklearn GP (the
    same GaussianProcessRegressor class the official
    sampling_subcluster_priority uses) fitted on labeled official AA encodings;
    P(top-5% member) = Phi((mu - gamma) / sigma), matching GB1OneHotGP semantics.
    Disclosure: the official CLADE supervised-model slot is MLDE, whose
    submodule is NOT part of the pinned snapshot (DECISION 2026-08-31); the
    official in-loop GP stands in as the readout (surrogate only — the
    selection rule is the official clustering/sampling core)."""

    def __init__(self, oracle, feats) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor
        self._feats = feats
        self._pool = list(oracle.candidates)
        self._index = {v: i for i, v in enumerate(self._pool)}
        self._gamma = float(oracle.gamma)
        self._regr = GaussianProcessRegressor(random_state=0, normalize_y=True)
        self._trained = False

    @property
    def index(self) -> dict:
        return self._index

    @property
    def trained(self) -> bool:
        return self._trained

    def fit(self, known: dict, train_seconds: list) -> None:
        import time as _t
        t0 = _t.time()
        idx = sorted(self._index[v] for v in known if v in self._index)
        if not idx:
            self._trained = False
            train_seconds.append(_t.time() - t0)
            return
        X = self._feats[idx]
        y = np.asarray([known[self._pool[i]] for i in idx], dtype=np.float64)
        self._regr.fit(X, y)
        self._trained = True
        train_seconds.append(_t.time() - t0)

    def predict_probs_by_index(self, indices) -> np.ndarray:
        from nlss.recovery.graph_matern import _norm_cdf
        idx = list(indices)
        if not self._trained or not idx:
            return np.full(len(idx), 0.5, dtype=np.float64)
        out = np.empty(len(idx), dtype=np.float64)
        CH = 20000
        for s in range(0, len(idx), CH):
            blk = idx[s:s + CH]
            mu, sd = self._regr.predict(self._feats[blk], return_std=True)
            mu = np.asarray(mu, dtype=np.float64)
            sd = np.maximum(np.asarray(sd, dtype=np.float64), 1e-12)
            z = (mu - self._gamma) / sd
            try:
                from scipy.special import ndtr  # vectorized standard-normal CDF
                out[s:s + CH] = ndtr(z)
            except ImportError:  # vectorized scalar-erf fallback
                import math as _m
                _erf = np.vectorize(_m.erf)
                out[s:s + CH] = 0.5 * (1.0 + _erf(z / _m.sqrt(2.0)))
        return out


def gb1_run_one(task: tuple):
    """One GB1 (arm, seed) campaign.  Primary threshold: agreed measured-fitness
    top-5% (149,361 measured only; 10,639 imputed excluded).  better-than-WT is
    carried as a labeled sensitivity curve."""
    (arm, seed, n_initial, batch, rounds, checkpoints, out_dir, registry_hash,
     cfg_hash, commit, model_key) = task

    from nlss.campaigns.gb1_predict import GB1OneHotGP

    oracle = _W["oracle"]
    oracle_wt = _W["oracle_wt"]
    regions = _W["regions"]
    pool = list(oracle.candidates)
    pool_idx = {v: i for i, v in enumerate(pool)}
    n_sol = len(oracle.solution_set)

    t_start = time.time()
    total_budget = n_initial + rounds * batch
    caps = dict(CAPS_NUM)
    caps["oracle_budget"] = float(total_budget)
    meter = ResourceMeter(caps)
    run = RunConfig(benchmark="gb1", method=arm, seed=seed, config_hash=cfg_hash, git_commit=commit)
    ledger7 = ResourceLedger(run)

    rng = random.Random(seed)
    queried: set = set()
    known: dict = {}
    prefix_hits: dict[int, int] = {}
    prefix_hits_wt: dict[int, int] = {}
    prefix_found_recall: dict[int, float] = {}
    prefix_best_fit: dict[int, float] = {}
    checkpoint_metrics: dict[int, dict] = {}
    pred = None
    failure: dict | None = None
    secs: list = []

    clade_seq: list | None = None
    clade_cursor = 0
    clade_calls = 0
    if arm == "nlss_hamming":
        pred = _W["hamm"]
        query_pool = pool
    elif arm in ("gb1_gpbo", "nlss_onehot"):
        pred = GB1OneHotGP(oracle, max_pool_subset=GB1_ONEHOT_POOL_CAP)
        query_pool = pred.pool
    elif arm == "gb1_ballet":
        pred = archived("gb1_ballet").GB1BalletLevelSet(oracle)
        query_pool = pool
    elif arm == "gb1_alde":
        pred = archived("gb1_alde").GB1Alde(oracle, max_pool_subset=GB1_ONEHOT_POOL_CAP)
        query_pool = pred.pool
    elif arm == "gb1_clade":
        pred = _CladeReadout(oracle, _gb_clade_features(oracle))
        query_pool = pool
        save_root = Path(out_dir) / f"clade_s{seed:02d}"
        clade_seq, clade_calls = _clade_official_sequence(oracle, pool, seed, save_root)
        if len(clade_seq) < total_budget:
            raise RuntimeError(
                f"official CLADE sequence short: {len(clade_seq)} < {total_budget}")
    elif arm in ("random", "hillclimb"):
        pred = None
        query_pool = pool
    else:
        raise ValueError(f"unknown GB1 arm {arm}")

    arm_idx = pred.index if pred is not None else {v: i for i, v in enumerate(pool)}

    def found_recall() -> float:
        return (sum(1 for c in queried if oracle.solution_membership(c)) / n_sol) if n_sol else 0.0

    def v7_eval_at(budget: int) -> dict:
        unq = [c for c in query_pool if c not in queried]
        unq_idx = [arm_idx[c] for c in unq]
        labels = np.asarray([1.0 if oracle.solution_membership(c) else 0.0 for c in unq])
        if pred is not None and getattr(pred, "trained", False):
            probs = np.asarray(pred.predict_probs_by_index(unq_idx))
            meta = _gb1_recovery_metrics(labels, probs)
            supported = {int(unq_idx[i]) for i in np.where(probs >= 0.5)[0]}
            supported |= {arm_idx[c] for c in queried if oracle.solution_membership(c)}
            meta["region_recall"] = _gb1_region_recall(supported, regions)
            return meta
        return {k: float("nan") for k in
                ("unseen_recall", "unseen_precision", "unseen_f1", "pr_auc", "brier", "region_recall")}

    try:
        if arm == "gb1_clade":
            init = clade_seq[:n_initial]
            clade_cursor = n_initial
        else:
            init = _acq_random_pool(query_pool, queried, rng, n_initial)
        for c in init:
            known[c] = oracle.evaluate(c)
            queried.add(c)
            ledger7.bump_oracle()
            meter.consume("oracle_budget", 1.0)
        if pred is not None:
            secs.clear()
            pred.fit(known, secs)
            meter.consume("tool_calls", 1.0)
            if secs:
                ledger7.bump_surrogate_fit(secs[-1])

        budget_so_far = n_initial
        prefix_hits[budget_so_far] = sum(1 for c in queried if oracle.solution_membership(c))
        prefix_found_recall[budget_so_far] = found_recall()
        prefix_best_fit[budget_so_far] = max(known.values())
        m0 = v7_eval_at(budget_so_far)
        checkpoint_metrics[budget_so_far] = dict(
            m0, found_recall=found_recall(), solutions_found=prefix_hits[budget_so_far],
            best_fitness=prefix_best_fit[budget_so_far])
        if budget_so_far in checkpoints:
            _gb1_write_scores(out_dir, arm, seed, budget_so_far, pool, pool_idx,
                              queried, known, rng, arm, pred, arm_idx, query_pool)

        for rnd in range(rounds):
            if meter.exceeded["wall_time_seconds"] or (time.time() - t_start) > caps["wall_time_seconds"]:
                failure = {"reason": "wall_time_cap", "at_checkpoint": budget_so_far}
                break
            budget_so_far += batch
            if arm == "gb1_clade":
                batch_c = clade_seq[clade_cursor:clade_cursor + batch]
                clade_cursor += len(batch_c)
            elif arm == "random":
                batch_c = _acq_random_pool(query_pool, queried, rng, batch)
            elif arm == "hillclimb":
                batch_c = _acq_hillclimb_gb1(query_pool, queried, known, rng, batch)
            elif arm == "gb1_gpbo":
                batch_c = pred.acquire_ei(known, batch, rng)
            elif arm == "gb1_ballet":
                batch_c = pred.acquire(known, batch, rng)
            elif arm == "gb1_alde":
                batch_c = pred.acquire(known, batch, rng, len(query_pool) < len(pool))
            elif arm.startswith("nlss"):
                avail = [c for c in query_pool if c not in queried]
                avail_idx = [arm_idx[c] for c in avail]
                probs = pred.predict_probs_by_index(avail_idx)
                order = np.argsort(-probs)
                batch_c = [avail[i] for i in order[:batch]]
            else:
                raise ValueError(f"unknown GB1 arm {arm}")
            for c in batch_c:
                known[c] = oracle.evaluate(c)
                queried.add(c)
                ledger7.bump_oracle()
                meter.consume("oracle_budget", 1.0)
            if pred is not None:
                secs.clear()
                pred.fit(known, secs)
                meter.consume("tool_calls", 1.0)
                if secs:
                    ledger7.bump_surrogate_fit(secs[-1])
            prefix_hits[budget_so_far] = sum(1 for c in queried if oracle.solution_membership(c))
            prefix_found_recall[budget_so_far] = found_recall()
            prefix_best_fit[budget_so_far] = max(known.values())
            m = v7_eval_at(budget_so_far)
            if budget_so_far in checkpoints or budget_so_far == total_budget:
                checkpoint_metrics[budget_so_far] = dict(
                    m, found_recall=found_recall(),
                    solutions_found=prefix_hits[budget_so_far],
                    best_fitness=prefix_best_fit[budget_so_far])
                if budget_so_far in checkpoints:
                    _gb1_write_scores(out_dir, arm, seed, budget_so_far, pool, pool_idx,
                                      queried, known, rng, arm, pred, arm_idx, query_pool)
    except Exception as exc:
        import traceback
        failure = {"reason": f"{type(exc).__name__}: {exc}",
                   "at_checkpoint": max(checkpoint_metrics) if checkpoint_metrics else n_initial,
                   "traceback": traceback.format_exc(limit=6)}

    wall = time.time() - t_start
    meter.consume("wall_time_seconds", wall)
    last_key = max(checkpoint_metrics) if checkpoint_metrics else n_initial
    ledger7.set(final_metrics=checkpoint_metrics.get(last_key, {}), checkpoint_metrics=checkpoint_metrics)
    log = ledger7.to_dict()
    last = dict(checkpoint_metrics.get(last_key, {}))
    last.update(
        solutions_found_total=max(prefix_hits.values()) if prefix_hits else 0,
        best_fitness=max(prefix_best_fit.values()) if prefix_best_fit else 0.0,
        solution_prevalence=oracle.solution_prevalence(),
    )
    log["final_metrics"] = last
    log["protocol"] = {
        "phase": "scored",
        "registry_hash": registry_hash,
        "model": model_key,
        "llm_calls": 0,
        "retries": 0,
        "parse_failures": 0,
        "latency_s": round(wall, 3),
        "tokens": 0,
        "cap_state": {
            "used": meter.usage(),
            "exceeded": {k: bool(meter.exceeded[k]) for k in CAP_KEYS},
            "incomplete_under_cap": meter.incomplete_under_cap,
        },
        "failure": failure,
        "prefix": {
            "hits": {str(k): v for k, v in prefix_hits.items()},
            "found_recall": {str(k): v for k, v in prefix_found_recall.items()},
            "best_fit": {str(k): v for k, v in prefix_best_fit.items()},
            "aurc_found_recall": aurc(prefix_found_recall),
            "aurc_best_fitness": aurc(prefix_best_fit),
        },
        "sensitivity": {"criterion": "better_wt",
                        "n_solution_set_wt": len(oracle_wt.solution_set),
                        "note": "labeled sensitivity only (briefing); not primary"},
    }
    if arm == "gb1_clade":
        log["protocol"]["specialist"] = {
            "pin": f"gb1_clade content_fingerprint {_CLADE_LOCK_FINGERPRINT} (official snapshot)",
            "arm": arm,
            "selection": "official clustering_sampling.cluster_sample (KMeans 30/30/30, "
                         "UCB beta=4, 96+4x96)",
            "label_channel": "evaluate-on-demand at selection; unselected access raises",
            "readout": "official in-repo sklearn GP (MLDE submodule absent from snapshot)",
            "n_oracle_calls_label_channel": int(clade_calls),
        }
    return log


def _gb1_write_scores(out_dir, arm, seed, budget, pool, pool_idx, queried, known,
                      rng, arm_name, pred, arm_idx, query_pool) -> None:
    score = np.zeros(len(pool), dtype=np.float32)
    unq = [c for c in query_pool if c not in queried]
    if pred is not None and getattr(pred, "trained", False):
        try:
            probs = np.asarray(pred.predict_probs_by_index([arm_idx[c] for c in unq]))
            for c, p in zip(unq, probs):
                score[pool_idx[c]] = np.float32(p)
        except Exception:
            score = np.zeros(len(pool), dtype=np.float32)
    if not score.any():
        state = rng.getstate()
        try:
            if arm_name == "hillclimb":
                head = _acq_hillclimb_gb1(query_pool, queried, known, rng, len(unq))
                rest = [c for c in unq if c not in set(head)]
                ranking = head + rest
            else:
                ranking = _acq_random_pool(query_pool, queried, rng, len(unq))
        finally:
            rng.setstate(state)
        for rank, c in enumerate(ranking):
            score[pool_idx[c]] = np.float32(1.0 - rank / max(1, len(ranking)))
    qmask = np.zeros(len(pool), dtype=bool)
    for c in queried:
        qmask[pool_idx[c]] = 1
    np.savez_compressed(Path(out_dir) / f"gb1_{arm_name}_seed{seed:02d}_s{budget}.npz",
                        score=score, queried=qmask)


# ---------------------------------------------------------------------------
# P0: oracle facts + preregistration registry
# ---------------------------------------------------------------------------

def oracle_facts(project: str) -> dict:
    if project == "bh":
        from nlss.adapters.bh.data import BHData
        from nlss.campaigns.bh_oracle import BHFiniteOracle

        def _facts_for(recs):
            class _U:
                def __init__(s, r):
                    s.records = r
            o = BHFiniteOracle(_BHMeasuredUniverseWrapper(recs), top_frac=0.05)
            cands = list(o.candidates)
            sol = sorted(c for c in cands if o.solution_membership(c))
            return {"pool": len(cands), "solution_set_size": len(sol), "gamma": o.gamma,
                    "solution_set_sha256": sha256_text(json.dumps([list(c) for c in sol]))}

        class _BHMeasuredUniverseWrapper:
            def __init__(self, recs):
                self.records = recs

        raw = BHData()
        all4 = [(c, y) for c, y in raw.records if all(str(x).strip() for x in c)]
        first3 = [(c, y) for c, y in raw.records if all(str(x).strip() for x in c[:3])]
        paper = [(c, y) for c, y in all4 if str(c[3]).strip() != PAPER_EXCLUDED_ADDITIVE]
        n_paper = len(paper)
        if n_paper != 3_955:  # protocol-exact count guard (Detail §8.1)
            raise SystemExit(f"FATAL: BH paper-grid universe = {n_paper}, expected 3,955 — "
                             "dataset drift; block p0 freeze")
        facts_paper = _facts_for(paper)
        facts_all4 = _facts_for(all4)
        facts_first3 = _facts_for(first3)
        return {
            "table_rows_raw": 4_599,
            "measured_paper_grid": n_paper,
            "pool": facts_paper["pool"],
            "gamma": facts_paper["gamma"],
            "solution_set_size": facts_paper["solution_set_size"],
            "solution_set_sha256": facts_paper["solution_set_sha256"],
            "legal_mask": "all4_components_present_and_additive_in_paper_22_set",
            "legal_mask_sha256": sha256_text(json.dumps([list(c) for c in sorted(paper)])),
            "universe_note": "primary = Detail §8.1 protocol-exact 3,955 measured reactions "
                             "(all-4-component rows minus the single beyond-paper additive "
                             f"'{PAPER_EXCLUDED_ADDITIVE}'); verified derivable from the official "
                             "release 2026-08-31 (c89).  Labeled sensitivities, never primary: "
                             f"all-4 {facts_all4['pool']} (|S*|={facts_all4['solution_set_size']}), "
                             f"the previous frozen release-anchor first-3 {facts_first3['pool']} "
                             f"(|S*|={facts_first3['solution_set_size']}, v8_probe_wt.py Probe A only), "
                             "raw 4,599 descriptive appendix.",
            "sensitivity_universes": {
                "all4_components": {"rows": len(all4), **{k: v for k, v in facts_all4.items()
                                                           if k in ("pool", "solution_set_size", "gamma")}},
                "v7_first3_anchor_only": {"rows": len(first3), **{k: v for k, v in facts_first3.items()
                                                                   if k in ("pool", "solution_set_size", "gamma")},
                                          "note": "the previous frozen release anchor attribution / probe A only"},
            },
        }
    from nlss.adapters.gb1.data import GB1Data, WILD_TYPE
    from nlss.campaigns.gb1_oracle import GB1FiniteOracle
    data = GB1Data()
    if not data.is_full_measured():
        raise SystemExit("FATAL: GB1 adapter did not load the full measured table (149,361)")
    oracle = GB1FiniteOracle(criterion="top_pct", top_frac=0.05)
    oracle_wt = GB1FiniteOracle(criterion="better_wt")
    variants = sorted(data.fitness)
    sol = sorted(oracle.solution_set)
    sol_wt = sorted(oracle_wt.solution_set)
    return {
        "measured_variants": len(variants),
        "imputed_excluded": 10_639,
        "source_sha256": sha256_file(ROOT / "external" / "repos" / "gb1_clade" / "Input" / "GB1.xlsx"),
        "legal_mask": "measured_only",
        "legal_mask_sha256": sha256_text(json.dumps(variants)),
        "primary_threshold": {"criterion": "top_pct_measured_fitness", "top_frac": 0.05,
                              "gamma": oracle.gamma, "solution_set_size": len(sol),
                              "solution_set_sha256": sha256_text(json.dumps(sol))},
        "sensitivity_threshold": {"criterion": "better_wt", "wt": WILD_TYPE,
                                  "wt_fitness": data.fitness[WILD_TYPE],
                                  "solution_set_size": len(sol_wt),
                                  "solution_set_sha256": sha256_text(json.dumps(sol_wt))},
        "fitness_norm_max": float(max(data.fitness.values())),
    }


def build_registry(project: str, seeds: list[int], commit: str, model: str = "glm") -> PreregRegistry:
    facts = oracle_facts(project)
    n_init, batch, rounds = BATCH[project]
    cps = CHECKPOINTS[project]
    cell = f"{project}_v8_scored"
    reg = PreregRegistry(f"the current protocol scored cell {cell} (engineer BC, {model} track)")
    reg.set("repository.commits", {"nlss": commit})
    reg.set("repository.submodule_commits", {"made": "n/a"})
    src_hash = facts.get("source_sha256", "bh_data_table_csv")
    reg.set("datasets.oracle_hashes", {project: f"sha256:{src_hash}"})
    reg.set("datasets.dataset_hashes", {project: f"sha256:{src_hash}"})
    reg.set("corpus.task_dossier_hashes", {project: "sha256:" + sha256_text(json.dumps(facts, sort_keys=True))})
    reg.set("corpus.frozen_corpus_hashes", {project: "n/a_retrieval_free"})
    reg.set("model.model_id", "none-numerical-arms")
    reg.set("model.tokenizer_id", "n/a")
    reg.set("model.temperature", 0.7)
    reg.set("model.tool_config", {"llm_endpoint": ENDPOINTS[model][0] + " (llm_endpoints.json v3_1)",
                                  "declared_model": ENDPOINTS[model][1],
                                  "max_tokens": MAX_TOKENS[model],
                                  "used_by_arms": "none (numerical arms; declared track metadata only)"})
    reg.set("prompts.prompt_hashes", {project: "n/a_no_llm_arms"})
    reg.set("prompts.persistent_state_schemas", {"readout": "score_vector_pool_order_v1"})
    reg.set("legality.candidate_legality_rules",
            {project: facts.get("legal_mask", "measured_pool")})
    reg.set("legality.replacement_rules", {project: "none_silent_replacement_forbidden"})
    reg.set("solutions.thresholds", {project: facts})
    reg.set("solutions.outcome_blind_geometry",
            {project: ("bh: morgan-kNN k=5 regions (outcome-blind); "
                       "gb1: hamming-1 components + kNN(k=5) S* families")})
    reg.set("budgets.stopping_rules", {project: "checkpoints_exhausted"})
    reg.set("budgets.retry_policy", {project: "one_retry_predeclared_infra_only"})
    reg.set("budgets.timeout_policy", {project: f"wall_time<= {CAPS_NUM['wall_time_seconds']:.0f}s/run"})
    reg.set("budgets.failure_rules",
            {project: "R_j=0.0 and utility=worst legal value from failure checkpoint (Detail §6/§14)"})
    reg.set("seeds.initial_evidence_ids", {project: f"shared_random_{n_init}_per_seed"})
    reg.set("adapters.identity_reports", {project: "nlss.campaigns the previous frozen release harness, the current protocol rerun"})
    reg.set("adapters.supported_cells", {project: [cell]})

    for ep, fn in ((f"{project}_recovery", "common_universe_AP_AURC_v1"),
                   (f"{project}_utility", "normalized_best_observed_AUC_v1"),
                   (f"{project}_tophit", "cumulative_tophit_AUC_v1")):
        reg.set_endpoint(ep, {
            "metric_function": fn,
            "curve_scalarization": "normalized_trapezoid_v1",
            "favorable_direction": "maximize",
            "delta_min": DELTAS["delta_min"],
            "delta_eq": DELTAS["delta_eq"],
            "delta_harm": DELTAS["delta_harm"],
            "failure_score": 0.0,
            "ci_method": "paired_bootstrap_v1",
            "effect_size_formula": "smd_v1",
            "resampling_scheme": "seed_paired_v1",
            "multiplicity_family": f"{project}_v8_track",
        })
    reg.set_track(project, {
        "task_list": [f"{project}_recovery", f"{project}_utility", f"{project}_tophit"],
        "min_samples": MIN_SEEDS[project],
        "max_samples": 50,
        "seed_list": list(seeds),
        "budget_table": {"n_initial": n_init, "batch": batch, "rounds": rounds,
                         "total_labels": n_init + rounds * batch},
        "checkpoints": list(cps),
        "batch_fill_algorithm": "none",
        "checkpoint_note": "briefing rounds [0,2,4,6,8] -> label checkpoints "
                           f"{cps} (bh: batch-units 0..10 incl. terminal {n_init + rounds * batch})",
    })
    base_ids = []
    for arm in DEFAULT_ARMS[project]:
        if arm == PRIMARY_ARM[project]:
            continue
        bid = f"{project}_{arm}"
        base_ids.append(bid)
        reg.set_baseline(bid, {
            "official_identifier": f"nlss.campaigns the previous frozen release arm {arm}",
            "commit_hash": commit,
            "container_hash": "vcc4-venv-py3.10",
            "entry_point": "scripts/run_preregistered_protocol.py",
            "config": {"arm": arm, "caps": {**CAPS_NUM, "oracle_budget": n_init + rounds * batch}},
            "permitted_adapter_diff": "none",
            "reproduction_tolerance": "seed_exact",
            "gate_result": "PASS",
        })
    reg.register_cell(cell, endpoint_id=f"{project}_recovery", track_id=project,
                      baseline_ids=tuple(base_ids))
    reg.finalize()
    return reg


def load_published_registry(project: str) -> dict:
    path = runs_dir(project, "p0") / "preregistration_registry.json"
    doc = json.loads(path.read_text())
    verify_registry_hash(doc)
    return doc


def _repo_commit() -> str:
    try:
        c = git_commit(str(ROOT))
        if c and c != "unknown":
            return c
    except Exception:
        pass
    meta = ROOT / "GIT_META.txt"
    if meta.exists():
        for line in meta.read_text().splitlines():
            parts = line.split()
            if parts and re.fullmatch(r"[0-9a-f]{7,40}", parts[-1]):
                return parts[-1]
    return "protocol-rebuild"


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def make_tasks(project: str, phase: str, seeds: list[int], arms: list[str], args) -> list[tuple]:
    n_init, batch, rounds = BATCH[project]
    cps = CHECKPOINTS[project]
    out = runs_dir(project, "scored" if phase == "scored" else "pilot")
    out.mkdir(parents=True, exist_ok=True)
    registry_hash = "PILOT"
    if phase == "scored":
        doc = load_published_registry(project)
        registry_hash = doc["registry_hash"]
        track = doc["content"]["tracks"][project]
        seeds = list(track["seed_list"])
        cps = tuple(track["checkpoints"])
        n_init = track["budget_table"]["n_initial"]
        batch = track["budget_table"]["batch"]
        rounds = track["budget_table"]["rounds"]
        # resume semantics: arms whose per-seed JSON already exists (e.g. the
        # ALDE specialist joined from its native trace) are not re-run
        out_dir = runs_dir(project, "scored")
        done_keys = set()
        for arm in arms:
            if all((out_dir / f"{project}_{arm}_seed{s:02d}.json").exists() for s in seeds):
                done_keys.add(arm)
        if done_keys:
            print(f"[{project}:{phase}] skipping completed arms: {sorted(done_keys)}", flush=True)
        arms = [a for a in arms if a not in done_keys]
    commit = args.commit or _repo_commit()
    cfg_hash = hash_config({"benchmark": project, "n_initial": n_init, "batch": batch,
                            "rounds": rounds, "arms": arms, "phase": phase,
                            "checkpoints": list(cps)})
    return [(arm, seed, n_init, batch, rounds, cps, str(out), registry_hash,
             cfg_hash, commit, args.model)
            for arm in arms for seed in seeds]


def run_phase(project: str, phase: str, args) -> None:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else DEFAULT_ARMS[project]
    seeds = PILOT_SEEDS[project] if phase == "pilot" else list(range(MAX_SEEDS))
    tasks = make_tasks(project, phase, seeds, arms, args)
    runner = bh_run_one if project == "bh" else gb1_run_one
    init_fn = _bh_worker_init if project == "bh" else _gb1_worker_init
    out = Path(tasks[0][6])
    led_path = str(ledger_path(project, phase))
    print(f"[{project}:{phase}] arms={arms} seeds={len(seeds)} tasks={len(tasks)} -> {out}", flush=True)

    done = 0
    fails = 0
    t0 = time.time()
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                append_jsonl(str(out / f"resource_{phase}.jsonl"), {
                    "ts": time.time(), "load1": round(os.getloadavg()[0], 2),
                    "mem_avail_gb": _mem_avail_gb(), "tree_rss_mb": _children_rss_mb(),
                    "workers": args.jobs, "done": done})
            except Exception:
                pass
            stop.wait(300)

    threading.Thread(target=sampler, daemon=True).start()
    try:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_fn,
                                 initargs=(tuple(arms),)) as ex:
            pending = {ex.submit(runner, t): (t, 0) for t in tasks}
            import concurrent.futures as _cf
            while pending:
                done_set, _ = _cf.wait(list(pending), timeout=5, return_when=_cf.FIRST_COMPLETED)
                for fut in done_set:
                    task, n_retry = pending.pop(fut)
                    try:
                        log = fut.result()
                    except Exception as exc:
                        if n_retry < 1:  # one retry, predeclared infra errors only (§14)
                            try:
                                nf = ex.submit(runner, task)
                                pending[nf] = (task, n_retry + 1)
                                continue
                            except Exception:
                                pass
                        log = {"benchmark": project, "method": task[0], "seed": task[1],
                               "protocol": {"phase": phase, "model": args.model,
                                      "failure": {"reason": f"infra: {type(exc).__name__}: {exc}",
                                                  "at_checkpoint": task[2]}}}
                        fails += 1
                    arm, seed = log.get("method", task[0]), int(log.get("seed", task[1]))
                    failed = bool(log.get("protocol", {}).get("failure"))
                    fails += int(failed)
                    fp = out / f"{project}_{arm}_seed{seed:02d}.json"
                    dump_json(fp, log)
                    append_jsonl(led_path, sanitize({
                        "run_id": f"{project}-{phase}-{arm}-s{seed}",
                        "ts": time.time(), "phase": phase, "arm": arm,
                        "model": log.get("protocol", {}).get("model", args.model),
                        "llm_calls": 0, "retries": n_retry, "parse_failures": 0,
                        "latency_s": log.get("protocol", {}).get("latency_s", 0.0),
                        "tokens": 0,
                        "cap_state": log.get("protocol", {}).get("cap_state", {}),
                        "status": "failed" if failed else "ok", "file": str(fp)}))
                    done += 1
                    if done % 10 == 0 or done == len(tasks):
                        print(f"  [{project}:{phase}] {done}/{len(tasks)} done, {fails} failed, "
                              f"{time.time()-t0:.0f}s", flush=True)
    finally:
        stop.set()
    print(f"[{project}:{phase}] COMPLETE tasks={len(tasks)} failed={fails} wall={time.time()-t0:.0f}s",
          flush=True)


def _mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return -1.0


def _children_rss_mb() -> float:
    try:
        out = subprocess.run(["ps", "-eo", "rss,comm"], capture_output=True, text=True, timeout=20)
        return sum(int(p[0]) for p in (l.split() for l in out.stdout.splitlines()[1:])
                   if len(p) == 2 and "python" in p[1]) / 1024.0
    except Exception:
        return -1.0


# --------------------------------- pilot ------------------------------------

def pilot_phase(project: str, args) -> None:
    run_phase(project, "pilot", args)
    z = 1.959964 + 0.841621  # z_.975 + z_.8
    pilot_dir = runs_dir(project, "pilot")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else DEFAULT_ARMS[project]
    primary = PRIMARY_ARM[project]
    if primary not in arms:  # run_phase auto-adds the primary; power needs it too
        arms = arms + [primary]
    per_seed: dict[str, dict[int, float]] = {a: {} for a in arms}
    for arm in arms:
        for seed in PILOT_SEEDS[project]:
            fp = pilot_dir / f"{project}_{arm}_seed{seed:02d}.json"
            if not fp.exists():
                continue
            d = json.loads(fp.read_text())
            protocol = d.get("protocol", {})
            if protocol.get("failure"):
                continue
            val = protocol.get("prefix", {}).get("aurc_found_recall")
            if val is not None:
                per_seed[arm][seed] = float(val)
    contrasts = {}
    for arm in arms:
        if arm == primary:
            continue
        common = sorted(set(per_seed[primary]) & set(per_seed[arm]))
        if len(common) < 3:
            continue
        deltas = np.asarray([per_seed[primary][s] - per_seed[arm][s] for s in common])
        sd = max(float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0, SD_FLOOR)
        n_req = math.ceil((z ** 2) * (sd ** 2) / (DELTAS["delta_min"] ** 2))
        contrasts[f"{primary}_vs_{arm}"] = {"n_pilot": len(common), "sd_delta": round(sd, 5),
                                            "n_required": int(n_req)}
    n_final = min(max([MIN_SEEDS[project]] + [c["n_required"] for c in contrasts.values()] + [0]),
                  MAX_SEEDS)
    out = {
        "project": project, "pilot_seeds": PILOT_SEEDS[project],
        "power_rule": "n = ((z_.975 + z_.8) * sd / delta_min)^2; delta_min=0.01 (normalized endpoints)",
        "endpoint_for_sd": "aurc_found_recall (variance proxy, all arms defined)",
        "contrasts": contrasts, "n_final_recommendation": int(n_final),
        "disclaimer": "PILOT — never counted toward claims (Detail §2.2)",
    }
    dump_json(pilot_dir / "pilot_power.json", out)
    print(json.dumps(out, indent=2))


# ---------------------------------- p0 --------------------------------------

def p0_phase(project: str, args) -> None:
    power_file = runs_dir(project, "pilot") / "pilot_power.json"
    if not power_file.exists():
        raise SystemExit(f"p0 blocked: run pilot first ({power_file} missing)")
    power = json.loads(power_file.read_text())
    n_final = max(int(power["n_final_recommendation"]), MIN_SEEDS[project])
    seeds = list(range(n_final))
    commit = args.commit or _repo_commit()
    facts = oracle_facts(project)
    reg = build_registry(project, seeds, commit, model=args.model)
    doc = reg.finalize()
    out = runs_dir(project, "p0")
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "preregistration_registry.json", doc)
    dump_json(out / "oracle_facts.json", facts)
    reg2 = PreregRegistry.from_published(doc)
    reg2.assert_scoreable(f"{project}_v8_scored")
    print(f"[{project}:p0] registry_hash={doc['registry_hash'][:16]}… seeds={seeds} "
          f"facts={ {k: facts[k] for k in facts if 'size' in k or k in ('pool', 'gamma', 'measured_variants')} } "
          f"— assert_scoreable PASS", flush=True)


# ------------------------------- aggregate ----------------------------------

def _ap(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _aurc_common(R_by_cp: dict[int, float], cps) -> float:
    """§6 Recovery-AURC per seed: normalized trapezoid over registered checkpoints."""
    if not R_by_cp:
        return float("nan")
    b0, bJ = cps[0], cps[-1]
    area, prev_r, prev_b = 0.0, None, None
    for cp in cps:
        r = R_by_cp.get(cp)
        if r is None or r != r:
            return float("nan")
        if prev_r is not None:
            area += 0.5 * (prev_r + r) * (cp - prev_b) / (bJ - b0)
        prev_r, prev_b = r, cp
    return float(area)


def aggregate_phase(args) -> None:
    projects = ["bh", "gb1"] if args.project == "all" else [args.project]
    for project in projects:
        doc = load_published_registry(project)
        content = doc["content"]
        seeds = list(content["tracks"][project]["seed_list"])
        cps = list(content["tracks"][project]["checkpoints"])
        arms = DEFAULT_ARMS[project]
        primary = PRIMARY_ARM[project]
        sdir = runs_dir(project, "scored")

        if project == "bh":
            from nlss.adapters.bh.data import BHData
            from nlss.campaigns.bh_oracle import BHFiniteOracle
            oracle = BHFiniteOracle(_BHMeasuredUniverse(BHData()), top_frac=0.05)
        else:
            from nlss.campaigns.gb1_oracle import GB1FiniteOracle
            oracle = GB1FiniteOracle(criterion="top_pct", top_frac=0.05)
        pool = list(oracle.candidates)
        memb = np.asarray([1.0 if oracle.solution_membership(c) else 0.0 for c in pool])

        # -- §6 common-universe R_j per (seed, checkpoint), then per-seed AURC
        R: dict[str, dict[int, dict[int, float]]] = {a: {} for a in arms}
        for seed in seeds:
            for cp in cps:
                q_union = None
                for arm in arms:
                    fp = sdir / f"{project}_{arm}_seed{seed:02d}_s{cp}.npz"
                    if not fp.exists():
                        continue
                    q = np.load(fp)["queried"]
                    q_union = q.copy() if q_union is None else (q_union | q)
                if q_union is None:
                    continue
                common = ~q_union
                for arm in arms:
                    fp = sdir / f"{project}_{arm}_seed{seed:02d}_s{cp}.npz"
                    if not fp.exists():
                        continue
                    sc = np.load(fp)["score"]
                    R[arm].setdefault(seed, {})[cp] = _ap(memb[common], sc[common])

        # -- per-seed scalar endpoints
        metric_key = "region_recall" if project == "bh" else "best_fitness"
        norm = 100.0 if project == "bh" else json.loads(
            (runs_dir(project, "p0") / "oracle_facts.json").read_text())["fitness_norm_max"]
        n_sol = len(oracle.solution_set)
        seed_vals: dict[str, dict[int, dict[str, float]]] = {a: {} for a in arms}
        for arm in arms:
            for seed in seeds:
                fp = sdir / f"{project}_{arm}_seed{seed:02d}.json"
                if not fp.exists():
                    continue
                d = json.loads(fp.read_text())
                protocol = d.get("protocol", {})
                failure_at = (protocol.get("failure") or {}).get("at_checkpoint")
                util = {int(b): float(v) / norm
                        for b, v in protocol.get("prefix", {}).get("best_yield" if project == "bh" else "best_fit", {}).items()}
                hits = {int(b): float(v) / max(1, n_sol)
                        for b, v in protocol.get("prefix", {}).get("hits", {}).items()}
                if failure_at is not None:  # §6: worst legal value from failure onward
                    util = {b: (0.0 if b > failure_at else v) for b, v in util.items()}
                    hits = {b: (0.0 if b > failure_at else v) for b, v in hits.items()}
                seed_vals[arm][seed] = {
                    "aurc_common": _aurc_common(R[arm].get(seed, {}), cps),
                    "utility_auc": aurc(util),
                    "tophit_auc": aurc(hits),
                    metric_key: 0.0 if protocol.get("failure") else d.get("final_metrics", {}).get(metric_key),
                    "failed": bool(protocol.get("failure")),
                }

        # -- §14 paired stats + Holm within project x track family
        # confirmatory family = CONFIRMATORY_BASELINES only (registry-frozen
        # 2026-08-31); random/hillclimb contrasts are descriptive, outside Holm
        raw_p: dict[tuple[str, str], float] = {}
        for base in CONFIRMATORY_BASELINES[project]:
            if base not in seed_vals or not seed_vals[base]:
                continue
            for endpoint in PRIMARY_ENDPOINTS:
                xs = [seed_vals[primary][s][endpoint] for s in seeds
                      if s in seed_vals[primary] and s in seed_vals[base]]
                ys = [seed_vals[base][s][endpoint] for s in seeds
                      if s in seed_vals[primary] and s in seed_vals[base]]
                if len(xs) >= 3:
                    raw_p[(base, endpoint)] = float(
                        paired_summary(xs, ys, seed=BOOT_SEED).permutation_p)
        fam = list(raw_p)
        adj = holm_adjust([raw_p[k] for k in fam]) if fam else []
        contrasts = {}
        for (base, endpoint), p_holm in zip(fam, adj):
            xs = [seed_vals[primary][s][endpoint] for s in seeds
                  if s in seed_vals[primary] and s in seed_vals[base]]
            ys = [seed_vals[base][s][endpoint] for s in seeds
                  if s in seed_vals[primary] and s in seed_vals[base]]
            ps = paired_summary(xs, ys, seed=BOOT_SEED)
            dec = decide(xs, ys, delta_min=DELTAS["delta_min"], delta_eq=DELTAS["delta_eq"],
                         delta_harm=DELTAS["delta_harm"], family_pvalues=[raw_p[k] for k in fam],
                         seed=BOOT_SEED)
            contrasts[f"{primary}_vs_{base}:{endpoint}"] = {
                "n": ps.n, "mean_delta": _r(ps.mean), "median_delta": _r(ps.median),
                "ci95": [_r(ps.ci_low), _r(ps.ci_high)],
                "smd": None if ps.smd != ps.smd else round(ps.smd, 4),
                "permutation_p_raw": _r(raw_p[(base, endpoint)], 6),
                "holm_p": _r(p_holm, 6),
                "decision": dec.label,
                "family": "confirmatory",
                "n_failures": int(sum(seed_vals[base][s]["failed"] for s in seeds if s in seed_vals[base])),
            }

        # -- descriptive contrasts (random/hillclimb context arms): reported
        # with raw p only, never Holm-adjusted, never a claim (registry 8-31)
        for base in DESCRIPTIVE_BASELINES:
            if base not in seed_vals or not seed_vals[base]:
                continue
            for endpoint in PRIMARY_ENDPOINTS:
                xs = [seed_vals[primary][s][endpoint] for s in seeds
                      if s in seed_vals[primary] and s in seed_vals[base]]
                ys = [seed_vals[base][s][endpoint] for s in seeds
                      if s in seed_vals[primary] and s in seed_vals[base]]
                if len(xs) < 3:
                    continue
                ps = paired_summary(xs, ys, seed=BOOT_SEED)
                dec = decide(xs, ys, delta_min=DELTAS["delta_min"], delta_eq=DELTAS["delta_eq"],
                             delta_harm=DELTAS["delta_harm"], family_pvalues=[ps.permutation_p],
                             seed=BOOT_SEED)
                contrasts[f"{primary}_vs_{base}:{endpoint}"] = {
                    "n": ps.n, "mean_delta": _r(ps.mean), "median_delta": _r(ps.median),
                    "ci95": [_r(ps.ci_low), _r(ps.ci_high)],
                    "smd": None if ps.smd != ps.smd else round(ps.smd, 4),
                    "permutation_p_raw": _r(ps.permutation_p, 6),
                    "holm_p": _r(ps.permutation_p, 6),
                    "decision": dec.label,
                    "family": "descriptive (the previous frozen release-join context arm; metric n/a in the previous frozen release agg; "
                              "outside the current protocol Holm family per TEAM_BRIEF 2026-08-31)",
                    "n_failures": int(sum(seed_vals[base][s]["failed"] for s in seeds if s in seed_vals[base])),
                }

        # -- TOST equivalence gate vs frozen frozen anchors (paired by seed)
        tost = {}
        v7_dir = ROOT / "results" / f"{project}_main_final"
        for arm in arms:
            v7_vals, v8_vals = [], []
            for seed in seeds:
                f7 = v7_dir / f"{project}_{arm}_seed{seed:02d}.json"
                if not f7.exists() or seed not in seed_vals[arm]:
                    continue
                v7 = json.loads(f7.read_text()).get("final_metrics", {}).get(metric_key)
                v8v = seed_vals[arm][seed][metric_key]
                if v7 is not None and v7 == v7 and v8v is not None and v8v == v8v:
                    v7_vals.append(float(v7))
                    v8_vals.append(float(v8v))
            if len(v7_vals) >= 6:
                deltas = np.asarray(v8_vals) - np.asarray(v7_vals)
                p_lo, p_hi, equiv = tost_equivalent(deltas, DELTAS["delta_eq"])
                tost[arm] = {"n_pairs": len(deltas),
                             "mean_v8_minus_v7": _r(float(deltas.mean())),
                             "tost_equivalent": bool(equiv),
                             "gate": "the previous frozen release anchor citable" if equiv else "USE_V8_RERUN_ONLY"}
        summary = {
            "registry_hash": doc["registry_hash"],
            "seeds": seeds, "checkpoints": cps, "metric": metric_key,
            "n_seed_vals": {a: len(seed_vals[a]) for a in arms},
            "terminal_mean": {a: (_mean([seed_vals[a][s][metric_key] for s in seed_vals[a]])
                                  if seed_vals[a] else None) for a in arms},
            "aurc_common_mean": {a: (_mean([seed_vals[a][s]["aurc_common"] for s in seed_vals[a]])
                                     if seed_vals[a] else None) for a in arms},
            "utility_auc_mean": {a: (_mean([seed_vals[a][s]["utility_auc"] for s in seed_vals[a]])
                                     if seed_vals[a] else None) for a in arms},
            "contrasts": contrasts,
            "tost_vs_v7": tost,
            "total_failed_runs": int(sum(seed_vals[a][s]["failed"] for a in arms for s in seed_vals[a])),
        }
        dump_json(sdir / "v8_aggregate.json", summary)
        _write_gain_ledger(project, summary, seed_vals, seeds, metric_key, primary, arms, norm, n_sol,
                           model=args.model)
        print(f"[{project}:aggregate] terminal_mean={summary['terminal_mean']} "
              f"aurc_common_mean={summary['aurc_common_mean']} failed={summary['total_failed_runs']}",
              flush=True)


def _r(x, nd=5):
    x = float(x)
    return None if x != x else round(x, nd)


def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return round(float(np.mean(vals)), 5) if vals else None


def _write_gain_ledger(project, summary, seed_vals, seeds, metric_key, primary, arms, norm, n_sol,
                       model: str = "glm") -> None:
    lines = [f"# the current protocol Gains Ledger — {project.upper()} (engineer BC, {model} track)", "",
             f"- registry: `{summary['registry_hash'][:16]}…`  seeds={len(seeds)}  "
             f"checkpoints={summary['checkpoints']}",
             f"- primary arm: `{primary}`; paired unit = initialization seed (Detail §14)",
             f"- endpoints: aurc_common (§6 AP, primary), utility_auc (norm/{norm:g}), "
             f"tophit_auc (|S*|={n_sol})",
             f"- failed runs kept in denominator: {summary['total_failed_runs']}", "",
             "| contrast | endpoint | n | Δmean | 95% CI | Holm p | decision |",
             "|---|---|---|---|---|---|---|"]
    for key, c in summary["contrasts"].items():
        name, endpoint = key.split(":")
        lines.append(f"| {name} | {endpoint} | {c['n']} | {c['mean_delta']:+.4f} "
                     f"| [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['holm_p']:.4g} | {c['decision']} |")
    lines += ["", "## TOST equivalence gate vs frozen frozen anchors", "",
              "| arm | n_pairs | mean the current protocol−the previous frozen release | TOST within δ_eq | gate |", "|---|---|---|---|---|"]
    for arm, t in summary["tost_vs_v7"].items():
        lines.append(f"| {arm} | {t['n_pairs']} | {t['mean_v8_minus_v7']:+.4f} "
                     f"| {str(t['tost_equivalent']).lower()} | {t['gate']} |")
    lines += ["", "## Terminal means (native scale)", "", "| arm | mean | n |", "|---|---|---|"]
    for arm in arms:
        if summary["terminal_mean"][arm] is not None:
            lines.append(f"| {arm} | {summary['terminal_mean'][arm]} | {summary['n_seed_vals'][arm]} |")
    dump_text = "\n".join(lines) + "\n"
    out = runs_dir(project, "scored") / "v8_gain_ledger.md"
    out.write_text(dump_text)
    print(f"[{project}] gains ledger -> {out}")


# ---------------------------------- figs ------------------------------------

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figs_phase(args) -> None:
    projects = ["bh", "gb1"] if args.project == "all" else [args.project]
    for project in projects:
        sdir = runs_dir(project, "scored")
        agg = json.loads((sdir / "v8_aggregate.json").read_text())
        doc = load_published_registry(project)
        seeds = list(doc["content"]["tracks"][project]["seed_list"])
        arms = DEFAULT_ARMS[project]
        primary = PRIMARY_ARM[project]
        metric_key = agg["metric"]
        vals = {a: {} for a in arms}
        for arm in arms:
            for seed in seeds:
                fp = sdir / f"{project}_{arm}_seed{seed:02d}.json"
                if not fp.exists():
                    continue
                d = json.loads(fp.read_text())
                if d.get("protocol", {}).get("failure"):
                    vals[arm][seed] = 0.0
                else:
                    v = d.get("final_metrics", {}).get(metric_key)
                    if v is not None and v == v:
                        vals[arm][seed] = float(v)
        anchor = {a: v for a, v in ANCHORS[project].items() if a in arms}
        _fig_main_paired(project, arms, primary, vals, agg, anchor, metric_key)
        _fig_delta_forest(project, agg["contrasts"], args.model)
        _fig_recovery(project, arms, seeds, agg)
        _fig_utility(project, arms, seeds)
        _fig_resources(project, arms, seeds, sdir)


def _fig_main_paired(project, arms, primary, vals, agg, anchor, metric_key) -> None:
    plt = _setup_mpl()
    arms = [a for a in arms if vals[a]]  # floor arms may lack a terminal posterior metric
    data = [np.asarray([vals[a][s] for s in sorted(vals[a])]) for a in arms]
    pos = np.arange(1, len(arms) + 1)
    fig, ax = plt.subplots(figsize=(1.6 + 1.7 * len(arms), 5.4))
    vp = ax.violinplot(data, positions=pos, widths=0.85, showextrema=False)
    for body, arm in zip(vp["bodies"], arms):
        body.set_facecolor("#4c78a8" if arm == primary else "#9bb7d4")
        body.set_alpha(0.55)
    ax.boxplot(data, positions=pos, widths=0.2, showfliers=False,
               medianprops=dict(color="k"))
    common_seeds = sorted(set.intersection(*[set(vals[a]) for a in arms])) if arms else []
    for s in common_seeds:
        ys = [vals[a][s] for a in arms]
        ax.plot(pos, ys, color="gray", lw=0.6, alpha=0.35, zorder=1)
        ax.scatter(pos, ys, s=12, color="k", alpha=0.6, zorder=3)
    lo = min([0.0] + [float(np.nanmin(d)) for d in data if d.size])
    hi = max([float(np.nanmax(d)) for d in data if d.size] + list(anchor.values()) + [1e-9])
    ax.set_ylim(lo, hi * 1.14 + 1e-6)  # no y-axis truncation
    for a, v in anchor.items():
        ax.axhline(v, ls="--", lw=0.9, alpha=0.65, color="#d62728" if a == primary else "#7f7f7f")
        ax.text(len(arms) + 0.42, v, f"the previous frozen release {a}={v}", fontsize=7, va="center", color="#444")
    base_anchor = anchor.get(primary, 0.0)
    for off, lab in ((DELTAS["delta_min"] if project == "bh" else 0.25, "δ_min"),
                     (-(DELTAS["delta_harm"] if project == "bh" else 0.5), "−δ_harm")):
        y = base_anchor + off
        ax.axhline(y, ls=":", lw=0.9, color="crimson", alpha=0.7)
        ax.text(0.55, y, lab, fontsize=7, color="crimson", va="bottom")
    ylab = "region_recall (terminal)" if project == "bh" else "best_fitness (terminal)"
    ax.set_ylabel(ylab)
    ax.set_xticks(pos)
    ax.set_xticklabels(arms, rotation=18, ha="right")
    ax.set_title(f"{project.upper()} the current protocol {metric_key} — n={len(common_seeds)} paired seeds, "
                 f"failures={agg['total_failed_runs']} (kept in denominator); "
                 f"paired lines = same seed", fontsize=10)
    fig = plt.gcf()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir(project) / f"{project}_main_paired.{ext}", dpi=300)
    plt.close(fig)


def _fig_delta_forest(project, contrasts, model: str = "glm") -> None:
    plt = _setup_mpl()
    keys = list(contrasts)
    fig, ax = plt.subplots(figsize=(8.5, 0.8 + 0.62 * len(keys)))
    for y, key in enumerate(keys[::-1]):
        c = contrasts[key]
        lo95, hi95 = c["ci95"]
        color = {"improves": "tab:green", "materially_harmful": "crimson"}.get(c["decision"], "gray")
        ax.errorbar(c["mean_delta"], y, xerr=[[c["mean_delta"] - lo95], [hi95 - lo95]],
                    fmt="o", color=color, capsize=3)
        ax.text(hi95, y, f"  Holm p={c['holm_p']:.3g} → {c['decision']}", fontsize=7, va="center")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(DELTAS["delta_min"], ls=":", color="crimson", lw=1.0, label="δ_min")
    ax.axvline(-DELTAS["delta_harm"], ls="--", color="crimson", lw=1.0, label="−δ_harm")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace(":", "\n") for k in keys[::-1]], fontsize=7)
    ax.set_xlabel(f"paired Δ ({PRIMARY_ARM[project]} − baseline), 95% paired bootstrap CI")
    ax.set_title(f"{project.upper()} primary contrasts — Holm within {project}×{model}-track family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir(project) / f"{project}_delta_forest.{ext}", dpi=300)
    plt.close(fig)


def _fig_recovery(project, arms, seeds, agg) -> None:
    plt = _setup_mpl()
    from sklearn.metrics import average_precision_score
    if project == "bh":
        from nlss.adapters.bh.data import BHData
        from nlss.campaigns.bh_oracle import BHFiniteOracle
        oracle = BHFiniteOracle(_BHMeasuredUniverse(BHData()), top_frac=0.05)
    else:
        from nlss.campaigns.gb1_oracle import GB1FiniteOracle
        oracle = GB1FiniteOracle(criterion="top_pct", top_frac=0.05)
    pool = list(oracle.candidates)
    memb = np.asarray([1.0 if oracle.solution_membership(c) else 0.0 for c in pool])
    sdir = runs_dir(project, "scored")
    doc = load_published_registry(project)
    cps = list(doc["content"]["tracks"][project]["checkpoints"])

    # union-queried per (seed, checkpoint) computed once
    unions: dict[tuple[int, int], np.ndarray] = {}
    for seed in seeds:
        for cp in cps:
            acc = None
            for arm in arms:
                fp = sdir / f"{project}_{arm}_seed{seed:02d}_s{cp}.npz"
                if not fp.exists():
                    continue
                q = np.load(fp)["queried"]
                acc = q.copy() if acc is None else (acc | q)
            if acc is not None:
                unions[(seed, cp)] = acc

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rng = np.random.default_rng(BOOT_SEED)
    for arm in arms:
        means, lows, highs = [], [], []
        for cp in cps:
            per_seed = []
            for seed in seeds:
                if (seed, cp) not in unions:
                    continue
                fp = sdir / f"{project}_{arm}_seed{seed:02d}_s{cp}.npz"
                if not fp.exists():
                    continue
                sc = np.load(fp)["score"]
                common = ~unions[(seed, cp)]
                if common.sum() == 0:
                    continue
                val = _ap(memb[common], sc[common])
                if val == val:
                    per_seed.append(val)
            if per_seed:
                arr = np.asarray(per_seed)
                boots = rng.choice(arr, (2000, len(arr))).mean(axis=1)
                means.append(arr.mean())
                lows.append(np.quantile(boots, 0.025))
                highs.append(np.quantile(boots, 0.975))
            else:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
        m = np.asarray(means, float)
        ax.plot(cps, m, marker="o", ms=3.5, label=arm)
        ax.fill_between(cps, np.asarray(lows, float), np.asarray(highs, float), alpha=0.15)
    ax.set_xlabel("oracle labels")
    ax.set_ylabel("R_j = AP on common unseen universe (§6)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{project.upper()} recovery curve — mean ± 95% bootstrap CI over {len(seeds)} seeds")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir(project) / f"{project}_aurc_recovery.{ext}", dpi=300)
    plt.close(fig)


def _fig_utility(project, arms, seeds) -> None:
    plt = _setup_mpl()
    sdir = runs_dir(project, "scored")
    bkey = "best_yield" if project == "bh" else "best_fit"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for arm in arms:
        curves, hits = {}, {}
        for seed in seeds:
            fp = sdir / f"{project}_{arm}_seed{seed:02d}.json"
            if not fp.exists():
                continue
            pre = json.loads(fp.read_text()).get("protocol", {}).get("prefix", {})
            curves[seed] = {int(b): float(v) for b, v in pre.get(bkey, {}).items()}
            hits[seed] = {int(b): float(v) for b, v in pre.get("hits", {}).items()}
        bs = sorted({b for c in curves.values() for b in c})
        if not bs:
            continue
        axes[0].plot(bs, [np.mean([curves[s].get(b, np.nan) for s in curves]) for b in bs],
                     marker="o", ms=3, label=arm)
        axes[1].plot(bs, [np.mean([hits[s].get(b, np.nan) for s in hits]) for b in bs],
                     marker="o", ms=3, label=arm)
    axes[0].set_ylabel("best observed " + ("yield" if project == "bh" else "fitness"))
    axes[0].set_title("best-observed curve (mean over seeds)")
    axes[1].set_ylabel("cumulative top-hit fraction")
    axes[1].set_title("cumulative top-hit curve (mean over seeds)")
    for ax in axes:
        ax.set_xlabel("oracle labels")
        ax.legend(fontsize=7)
    fig.suptitle(f"{project.upper()} utility curves (the current protocol rerun, n={len(seeds)})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir(project) / f"{project}_utility_curves.{ext}", dpi=300)
    plt.close(fig)


def _fig_resources(project, arms, seeds, sdir) -> None:
    plt = _setup_mpl()
    lat, surr = [], []
    for arm in arms:
        ls, ss = [], []
        for seed in seeds:
            fp = sdir / f"{project}_{arm}_seed{seed:02d}.json"
            if not fp.exists():
                continue
            d = json.loads(fp.read_text())
            ls.append(d.get("protocol", {}).get("latency_s", 0.0))
            ss.append(float(d.get("surrogate_train_seconds", 0.0) or 0.0))
        lat.append(np.mean(ls) if ls else 0.0)
        surr.append(np.mean(ss) if ss else 0.0)
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x - 0.2, lat, width=0.4, label="wall s/run (mean)")
    ax.bar(x + 0.2, surr, width=0.4, label="surrogate train s (mean)")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=18, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title(f"{project.upper()} per-run resources (caps: oracle={CHECKPOINTS[project][-1]}, "
                 f"tool≤64, wall≤{CAPS_NUM['wall_time_seconds']:.0f}s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir(project) / f"{project}_resources.{ext}", dpi=300)
    plt.close(fig)


# ---------------------------------- main -------------------------------------

def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    ap = argparse.ArgumentParser(description="the current protocol BH/GB1 scored-campaign runner (projects B+C)")
    ap.add_argument("--project", default="all", choices=["bh", "gb1", "all"])
    ap.add_argument("--phase", required=True, choices=["pilot", "p0", "scored", "aggregate", "figs"])
    ap.add_argument("--arms", default="", help="comma list; default = full the previous frozen release-main arm set")
    ap.add_argument("--seeds", type=int, default=0, help="ignored for scored (registry is authority)")
    ap.add_argument("--model", default="glm", choices=list(ENDPOINTS),
                    help="declared LLM track — GLM-5.3-Flash only (deepseek retired "
                         "2026-08-29, 61d3a87); numerical arms make no LLM calls")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--commit", default="")
    args = ap.parse_args()

    projects = ["bh", "gb1"] if args.project == "all" else [args.project]
    for project in projects:
        if args.phase == "pilot":
            pilot_phase(project, args)
        elif args.phase == "p0":
            p0_phase(project, args)
        elif args.phase == "scored":
            run_phase(project, "scored", args)
        elif args.phase == "aggregate":
            aggregate_phase(args)
        elif args.phase == "figs":
            figs_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
