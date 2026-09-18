"""GB1 protein-fitness campaign runner — reproducible recovery-vs-budget experiment.

Cross-domain replication of the BH protocol on the GB1 combinatorial
fitness landscape: seeded campaigns, recovery vs budget, AURC, and
best-fitness summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time as _time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from nlss.campaigns.base import (  # noqa: E402
    CampaignConfig,
    ResourceLedger,
    RunConfig,
    git_commit,
    hash_config,
)
from nlss.campaigns.gb1_oracle import GB1FiniteOracle  # noqa: E402
from nlss.campaigns.gb1_predict import GB1HammingProp, GB1OneHotGP  # noqa: E402
from nlss.campaigns.gb1_ballet import GB1BalletLevelSet  # noqa: E402
from nlss.campaigns.gb1_alde import GB1Alde  # noqa: E402
from nlss.campaigns.gb1_qd import GB1QDMAPElites  # noqa: E402
from nlss.campaigns.gb1_rankflow import GB1RankFlow  # noqa: E402
from nlss.adapters.gb1.data import GB1Data, WILD_TYPE  # noqa: E402
from nlss.llm.deepseek import DeepseekV4Flash  # noqa: E402

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _solution_regions(oracle: GB1FiniteOracle, hamm: GB1HammingProp) -> dict[int, list[int]]:
    """Connected components of the Hamming-1 graph induced on S* (frozen geometry).

    Returns {region_id: [pool_indices of members]}.  Outcome-blind geometry
    (variant identity only); yields decide membership only.
    """
    idx = hamm.index
    sol_idx = [idx[v] for v in oracle.solution_set]
    sol_set = set(sol_idx)
    nab = {i: set(hamm._neighbors.get(i, ())) for i in sol_idx}
    regions: list[list[int]] = []
    visited: set[int] = set()
    for start in sol_idx:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in nab[u]:
                if w in sol_set and w not in visited:
                    visited.add(w)
                    stack.append(w)
        regions.append(comp)
    return {rid: comp for rid, comp in enumerate(regions)}


def _recovery_metrics_array(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    region_members: dict[int, list[int]] | None = None,
    region_hits: set[int] | None = None,
    region_rank_K: int | None = None,
    unq_pool_idx: np.ndarray | None = None,
) -> dict:
    """Array-based recovery metrics over unqueried candidates ()."""
    n = len(labels)
    if n == 0:
        return {k: float("nan") for k in
                ("unseen_recall", "unseen_precision", "unseen_f1", "pr_auc", "brier", "region_recall")}
    pred_pos = probs >= 0.5
    tp = int(np.sum(pred_pos & (labels == 1)))
    fp = int(np.sum(pred_pos & (labels == 0)))
    fn = int(np.sum((~pred_pos) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # PR-AUC (average precision)
    order = np.argsort(-probs, kind="mergesort")
    rl = labels[order]
    pos_idx = np.where(rl == 1)[0]
    pr_auc = float(np.mean((np.arange(len(pos_idx)) + 1) / (pos_idx + 1))) if len(pos_idx) else float("nan")
    brier = float(np.mean((probs - labels) ** 2))
    # RegionRecall: fraction of true S* regions touched by recovery.  With
    # region_rank_K set, "touched" = the method's TOP-K ranked (by prob) unqueried
    # candidates (precision-immune, report 9.9); else the nominal p>=0.5 support.
    region_recall = float("nan")
    if region_members is not None:
        rhs = region_hits
        if region_rank_K is not None and unq_pool_idx is not None:
            order = np.argsort(-probs, kind="mergesort")
            rhs = set(int(unq_pool_idx[i]) for i in order[: min(region_rank_K, len(order))])
        if rhs:
            rhs_set = set(rhs)
            covered = 0
            for rid, members in region_members.items():
                if any(m in rhs_set for m in members):
                    covered += 1
            region_recall = covered / len(region_members) if region_members else float("nan")
    return {
        "unseen_recall": recall,
        "unseen_precision": precision,
        "unseen_f1": f1,
        "pr_auc": pr_auc,
        "brier": brier,
        "region_recall": region_recall,
    }


def _onehot(v):
    vec = np.zeros(4 * len("ACDEFGHIKLMNPQRSTVWY"), dtype=np.float64)
    for i, ch in enumerate(v):
        vec[i * 20 + "ACDEFGHIKLMNPQRSTVWY".index(ch)] = 1.0
    return vec


def _solution_regions_knn(oracle, k: int = 5) -> dict[int, list[int]]:
    """Finer, discriminating solution-family geometry over S* (report 9.5-style).

    The Hamming-1 c.c. over S* collapses to [3640,1,2] (one giant component), so
    RegionRecall over it is coarse.  Use k-nearest-neighbours on the 80-d one-hot
    (Tanimoto-free, Euclidean) over S* members -> many solution families.
    Outcome-blind (variant identity only); returns {rid: [full-pool index]}.
    """
    sol = list(oracle.solution_set)
    X = np.asarray([_onehot(v) for v in sol], dtype=np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    adj = [set() for _ in range(len(sol))]
    CH = 500
    for s0 in range(0, len(sol), CH):
        blk = Xn[s0:s0+CH]
        sim = blk @ Xn.T
        np.fill_diagonal(sim.max(axis=1) if False else [0.0]*len(sim), 0.0) if False else None
        for r in range(blk.shape[0]):
            gi = s0 + r
            sim[r][gi] = -1.0
            for j in np.argsort(-sim[r])[:k]:
                adj[gi].add(int(j)); adj[int(j)].add(gi)
    comp = [-1]*len(sol); cid = 0
    unvis = set(range(len(sol)))
    while unvis:
        start = unvis.pop(); stack=[start]; comp[start]=cid
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in unvis:
                    unvis.discard(v); comp[v]=cid; stack.append(v)
        cid += 1
    idx = {v: i for i, v in enumerate(oracle.candidates)}
    out = {}
    for rid in range(cid):
        members = [idx[sol[i]] for i in range(len(sol)) if comp[i] == rid]
        if members: out[rid] = members
    return out


def _aurc(recovery_by_budget: dict[int, float]) -> float:
    pairs = sorted((int(b), float(r)) for b, r in recovery_by_budget.items())
    if len(pairs) < 2 or pairs[-1][0] <= pairs[0][0]:
        return 0.0
    bmin, bmax = pairs[0][0], pairs[-1][0]
    area = sum(0.5 * (r0 + r1) * (b1 - b0) for (b0, r0), (b1, r1) in zip(pairs, pairs[1:]))
    return float(area / (bmax - bmin))


def _gb1_llm(known, pool_set, queried, n, strategy, rng):
    """LLM causal-control for GB1: propose 4-AA variant strings from observed history."""
    prov = DeepseekV4Flash(
        api_key=os.environ.get("NLSS_LLM_API_KEY", ""),
        base_url=os.environ.get("NLSS_LLM_BASE_URL"),
        model="deepseek-v4-flash", max_output_tokens=8192,
    )
    top = sorted(known.items(), key=lambda kv: -kv[1])[:40]
    if strategy == "scalar":
        ctx = "## Observed top GB1 variants and fitness (best-first)\n" + "\n".join(
            f"- {v}: {y:.3f}" for v, y in top)
    else:  # history
        seen = list(known.items())
        seen_sorted = sorted(seen, key=lambda kv: kv[1])
        n_mid = len(seen)//2
        ctx = "## Observed GB1 variants (chronological-ish sample, fitness)\n" + "\n".join(
            f"- {v}: {y:.3f}" for v, y in seen[: max(1, int(len(seen)*0.4))])
    if strategy == "scalar_div":
        ctx += ("\n\nDiversity directive: cover several distinct 4-site chemistries, "
                "not one scaffold.")
    sysp = ("You are a protein engineer. Propose {} distinct 4-amino-acid GB1 variants (one per "
            "line, uppercase AAs from ACDEFGHIKLMNPQRSTVWY, length 4) that you expect to exceed "
            "wild-type fitness given the history. Output ONLY the 4-AA strings.").format(max(1, int(n*1.2)))
    gen = None
    for attempt in range(3):  # backoff retry for flaky tunnel
        try:
            gen = prov.generate(sysp, ctx, temperature=0.7)
            break
        except Exception as e:
            if attempt == 2:
                print(f"[gb1_llm] give up after 3 tries ({type(e).__name__}), return random"); return _acquire_random(list(pool_set), set(queried), rng, n), 0
            time_sleep = 5 * (attempt + 1)
            import time; time.sleep(time_sleep)
    if gen is None:
        return _acquire_random(list(pool_set), set(queried), rng, n), 0
    out = []
    for line in gen.text.splitlines():
        v = "".join(ch for ch in line.strip() if ch in "ACDEFGHIKLMNPQRSTVWY")[:4]
        if len(v) == 4 and v in pool_set and v not in queried and v not in out:
            out.append(v)
        if len(out) >= n:
            break
    return out, getattr(gen, "input_tokens", 0) + getattr(gen, "output_tokens", 0)


def _acquire_random(query_pool, queried: set, rng, n: int) -> list:
    avail = [c for c in query_pool if c not in queried]
    rng.shuffle(avail)
    return avail[:n]


def _acquire_hillclimb(query_pool, queried: set, known: dict, rng, n: int) -> list:
    if not known:
        return _acquire_random(query_pool, queried, rng, n)
    best = max(known, key=lambda c: known[c])
    pool_set = set(query_pool)
    proposed = []
    for i in range(4):
        for aa in _AA:
            if aa != best[i]:
                cand = best[:i] + aa + best[i + 1:]
                if cand in pool_set and cand not in queried:
                    proposed.append(cand)
    rng.shuffle(proposed)
    return proposed[:n]


def _run_arm(oracle, arm, seed, cfg, ledger, hamm, regions, sol_idx_set, onehot_cap=None):
    rng = random.Random(seed)
    pool = list(oracle.candidates)
    n_sol = len(oracle.solution_set)
    queried: set = set()
    known: dict = {}
    prefix_hits: dict[int, int] = {}
    prefix_found_recall: dict[int, float] = {}
    prefix_unseen_recall: dict[int, float] = {}
    prefix_best_fit: dict[int, float] = {}
    checkpoint_metrics: dict[int, dict] = {}
    pred = None

    # The arm's pool/index: NLSS arms route through their predictor (the one-hot
    # control may be capped to a sampled subset); floor arms use the full pool.
    arm_idx = hamm.index
    arm_pool = pool
    is_subset = False

    # Determine the arm's query pool FIRST (predictor construction needs no
    # observations): NLSS arms route through their predictor (one-hot control may
    # be capped to a sampled subset); floor arms use the full pool.  The arm only
    # ever queries within its own pool (oracle parity within that scope).
    if arm == "nlss_hamming":
        pred = GB1HammingProp(oracle)
        arm_idx, arm_pool = pred.index, pred.pool
        is_subset = False
    elif arm in ("nlss_onehot", "gb1_gpbo"):
        pred = GB1OneHotGP(oracle, max_pool_subset=onehot_cap)
        arm_idx, arm_pool = pred.index, pred.pool
        is_subset = len(arm_pool) < len(pool)
    elif arm == "gb1_alde":
        pred = GB1Alde(oracle, max_pool_subset=onehot_cap)
        arm_idx, arm_pool = pred.index, pred.pool
        is_subset = len(arm_pool) < len(pool)
    elif arm == "gb1_qd":
        pred = GB1QDMAPElites(oracle)
        arm_idx, arm_pool = pred.index, pred.pool
        is_subset = False
    elif arm == "gb1_rankflow":
        pred = GB1RankFlow(oracle)
        arm_idx, arm_pool = pred.index, pred.pool
        is_subset = False
    elif arm == "gb1_ballet":
        pred = GB1BalletLevelSet(oracle)  # full pool (logistic is O(n*d), no subset limit)
        arm_idx, arm_pool = pred.index, pred.pool
        is_subset = False
    elif arm.startswith("gb1_llm"):
        pred = None
        arm_idx, arm_pool = hamm.index, pool
        is_subset = False
    else:
        pred = None
        arm_idx, arm_pool = hamm.index, pool
        is_subset = False
    query_pool = arm_pool

    def found_recall():
        found = sum(1 for c in queried if oracle.solution_membership(c))
        return found / n_sol if n_sol else 0.0

    def eval_at(budget):
        unq = [c for c in query_pool if c not in queried]
        unq_idx = [arm_idx[c] for c in unq]
        labels = np.asarray([1.0 if oracle.solution_membership(c) else 0.0 for c in unq])
        if (arm.startswith("nlss") or arm in ("gb1_gpbo", "gb1_ballet", "gb1_alde", "gb1_rankflow")) and pred is not None:
            probs = np.asarray(pred.predict_probs_by_index(unq_idx))
            supported = {unq_idx[i] for i in np.where(probs >= 0.5)[0]}
            supported |= {arm_idx[c] for c in queried if oracle.solution_membership(c)}
            if is_subset:
                meta = _recovery_metrics_array(labels, probs, region_members=None, region_hits=None)
                meta["region_recall"] = float("nan")
                return meta
            return _recovery_metrics_array(labels, probs, region_members=regions,
                                           region_hits=supported, region_rank_K=60,
                                           unq_pool_idx=np.asarray(unq_idx))
        # No posterior: unseen-recovery not applicable (only found-recall counts).
        return {k: float("nan") for k in
                ("unseen_recall", "unseen_precision", "unseen_f1", "pr_auc", "brier", "region_recall")}

    # initial 96 (within the arm's query pool)
    init = _acquire_random(query_pool, queried, rng, cfg.n_initial)
    for c in init:
        known[c] = oracle.evaluate(c)
        queried.add(c)
        ledger.bump_oracle()
    if arm.startswith("nlss") or arm in ("gb1_gpbo","gb1_ballet","gb1_alde","gb1_rankflow","gb1_qd"):
        secs = []
        pred.fit(known, secs)
        ledger.bump_surrogate_fit(secs[-1] if secs else 0.0)

    budget_so_far = cfg.n_initial
    prefix_hits[budget_so_far] = sum(1 for c in queried if oracle.solution_membership(c))
    prefix_found_recall[budget_so_far] = found_recall()
    prefix_best_fit[budget_so_far] = max(known.values())
    m0 = eval_at(budget_so_far)
    prefix_unseen_recall[budget_so_far] = m0["unseen_recall"]
    checkpoint_metrics[budget_so_far] = dict(m0, found_recall=found_recall(),
                                             solutions_found=prefix_hits[budget_so_far],
                                             best_fitness=prefix_best_fit[budget_so_far])

    for rnd in range(cfg.rounds):
        budget_so_far += cfg.batch_size
        if arm == "random":
            batch = _acquire_random(query_pool, queried, rng, cfg.batch_size)
        elif arm == "hillclimb":
            batch = _acquire_hillclimb(query_pool, queried, known, rng, cfg.batch_size)
        elif arm == "gb1_gpbo":
            batch = pred.acquire_ei(known, cfg.batch_size, rng)
        elif arm == "gb1_alde":
            batch = pred.acquire(known, cfg.batch_size, rng, is_subset)
        elif arm == "gb1_qd":
            batch = pred.acquire(known, cfg.batch_size, rng)
        elif arm == "gb1_rankflow":
            batch = pred.acquire(known, cfg.batch_size, rng)  # known dict (best_f + pool membership)
        elif arm == "gb1_ballet":
            batch = pred.acquire(known, cfg.batch_size, rng)
        elif arm.startswith("nlss"):
            avail = [c for c in query_pool if c not in queried]
            avail_idx = [arm_idx[c] for c in avail]
            probs = pred.predict_probs_by_index(avail_idx)
            order = np.argsort(-probs)
            batch = [avail[i] for i in order[: cfg.batch_size]]
        elif arm.startswith("gb1_llm"):
            strat = "scalar_div" if arm == "gb1_llm_scalar_div" else ("scalar" if arm == "gb1_llm_scalar" else "history")
            llm_batch, _tok = _gb1_llm(known, set(hamm.index) if False else set(pool), queried, cfg.batch_size, strat, rng)
            batch = llm_batch
            pad = [c for c in (arm_pool) if c not in queried and c not in set(batch)]
            rng.shuffle(pad)
            batch = batch + pad[: max(0, cfg.batch_size - len(batch))]
        else:
            raise ValueError(f"unknown arm {arm}")
        for c in batch:
            known[c] = oracle.evaluate(c)
            queried.add(c)
            ledger.bump_oracle()
        if arm.startswith("nlss") or arm in ("gb1_gpbo","gb1_ballet","gb1_alde","gb1_rankflow","gb1_qd"):
            secs = []
            pred.fit(known, secs)
            if secs:
                ledger.bump_surrogate_fit(secs[-1])
        prefix_hits[budget_so_far] = sum(1 for c in queried if oracle.solution_membership(c))
        prefix_found_recall[budget_so_far] = found_recall()
        prefix_best_fit[budget_so_far] = max(known.values())
        m = eval_at(budget_so_far)
        prefix_unseen_recall[budget_so_far] = m["unseen_recall"]
        if budget_so_far in cfg.checkpoints:
            checkpoint_metrics[budget_so_far] = dict(m, found_recall=found_recall(),
                                                     solutions_found=prefix_hits[budget_so_far],
                                                     best_fitness=prefix_best_fit[budget_so_far])

    last = checkpoint_metrics.get(max(checkpoint_metrics), m0)
    return {
        "prefix_hits": prefix_hits,
        "prefix_found_recall": prefix_found_recall,
        "prefix_unseen_recall": prefix_unseen_recall,
        "prefix_best_fit": prefix_best_fit,
        "checkpoint_metrics": checkpoint_metrics,
        "aurc_found_recall": _aurc(prefix_found_recall),
        "aurc_unseen_recall": _aurc(prefix_unseen_recall),
        "n_solutions_found_total": max(prefix_hits.values()) if prefix_hits else 0,
        "best_fitness_final": max(prefix_best_fit.values()) if prefix_best_fit else 0.0,
        "solution_prevalence": oracle.solution_prevalence(),
        "final_metrics": last,
        "n_solution_set": len(oracle.solution_set),
    }


def main():
    ap = argparse.ArgumentParser(description="GB1 recovery campaign runner")
    ap.add_argument("--arms", default="random,nlss_hamming,hillclimb")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-initial", type=int, default=96)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--criterion", default="better_wt")
    ap.add_argument("--out", default="results/gb1_campaign")
    ap.add_argument("--onehot-pool-cap", type=int, default=25000,
                    help="Candidate-pool cap for the nlss_onehot control arm "
                         "(full 149k is O(n_pool*n_train^2) and infeasible); "
                         "the hamming + floor arms always use the full pool.")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cfg = CampaignConfig(n_initial=args.n_initial, batch_size=args.batch, rounds=args.rounds,
                         checkpoints=tuple(args.n_initial + i * args.batch for i in range(1, args.rounds + 1)))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    commit = git_commit(repo_root)
    cfg_hash = hash_config({"benchmark": "gb1", "criterion": args.criterion,
                            "n_initial": args.n_initial, "batch": args.batch, "rounds": args.rounds,
                            "arms": arms})

    t0 = _time.time()
    oracle = GB1FiniteOracle(criterion=args.criterion)
    hamm = GB1HammingProp(oracle)
    regions = _solution_regions_knn(oracle, k=5)  # finer S* solution-family geometry
    sol_idx_set = set(hamm.index[v] for v in oracle.solution_set)
    print(f"[gb1-campaign] pool={len(oracle.candidates)} |S*|={len(oracle.solution_set)} "
          f"prev={oracle.solution_prevalence():.3f} gamma={oracle.gamma:.3f} "
          f"regions={[len(r) for r in regions.values()]} (geo {_time.time()-t0:.1f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    per_arm = {arm: [] for arm in arms}
    budget_cps = sorted(cfg.checkpoints)
    print("\n[gb1-campaign] cumulative #S* found (mean over seeds):")
    print("  " + "".join(f"{b:>8}" for b in budget_cps))
    for arm in arms:
        for seed in range(args.seeds):
            run = RunConfig(benchmark="gb1", method=arm, seed=seed, config_hash=cfg_hash, git_commit=commit)
            ledger = ResourceLedger(run)
            result = _run_arm(oracle, arm, seed, cfg, ledger, hamm, regions, sol_idx_set,
                              onehot_cap=args.onehot_pool_cap)
            ledger.set(final_metrics=result["final_metrics"], checkpoint_metrics=result["checkpoint_metrics"])
            log = ledger.to_dict()
            log["final_metrics"].update({"solutions_found_total": result["n_solutions_found_total"],
                                         "best_fitness": result["best_fitness_final"],
                                         "solution_prevalence": result["solution_prevalence"]})
            with open(os.path.join(args.out, f"gb1_{arm}_seed{seed:02d}.json"), "w") as f:
                json.dump(log, f, indent=2)
            per_arm[arm].append(result)
        rows = []
        for b in budget_cps:
            if b <= cfg.total_budget:
                vals = [r["prefix_hits"].get(b, 0) for r in per_arm[arm]]
                rows.append(f"{np.mean(vals):7.1f}±{np.std(vals):.1f}")
        print(f"  {arm:<16}" + "".join(rows), flush=True)

    print("\n[gb1-campaign] AURC (found-recall vs budget):")
    for arm in arms:
        vals = [r["aurc_found_recall"] for r in per_arm[arm]]
        print(f"  {arm:<16} AURC={np.mean(vals):.4f}±{np.std(vals):.4f}")
    print("[gb1-campaign] AURC (unseen-recall; posterior arms only):")
    for arm in arms:
        vals = [r["aurc_unseen_recall"] for r in per_arm[arm]]
        if all(np.isnan(v) for v in vals):
            print(f"  {arm:<16} n/a (no posterior)")
        else:
            print(f"  {arm:<16} AURC-unseen={np.nanmean(vals):.4f}±{np.nanstd(vals):.4f}")

    print("\n[gb1-campaign] final best-fitness attained (mean±sd):")
    for arm in arms:
        vals = [r["best_fitness_final"] for r in per_arm[arm]]
        print(f"  {arm:<16} BestFitness@480={np.mean(vals):.4f}±{np.std(vals):.4f}")

    print("\n[gb1-campaign] final recovery (mean±sd; unseen = posterior arms only):")
    keys = ["found_recall", "unseen_recall", "unseen_precision", "unseen_f1", "pr_auc", "region_recall", "brier"]
    print("  " + "".join(f"{k:>15}" for k in keys))
    for arm in arms:
        row = []
        for k in keys:
            vals = [r["final_metrics"].get(k, float("nan")) for r in per_arm[arm]]
            label = "n/a" if all(np.isnan(v) for v in vals) else f"{np.nanmean(vals):15.4f}"
            row.append(label)
        print(f"  {arm:<16}" + "".join(row), flush=True)

    print(f"\n[gb1-campaign] DONE ({_time.time()-t0:.0f}s) — logs under {args.out}/")


if __name__ == "__main__":
    main()
