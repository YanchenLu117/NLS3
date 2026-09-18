"""Buchwald–Hartwig (BH) campaign runner — reproducible recovery-vs-budget experiment.

Runs seeded optimization campaigns over the BH reaction-yield dataset and
reports recovery (found-recall of the top variants) against query budget,
plus AURC and best-yield summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from nlss.campaigns.base import (  # noqa: E402
    CampaignConfig,
    ResourceLedger,
    RunConfig,
    git_commit,
    hash_config,
)
from nlss.campaigns.bh_gp import BHPoolGP  # noqa: E402
from nlss.campaigns.bh_graph import BHGraphProp  # noqa: E402
from nlss.campaigns.bh_oracle import BHFiniteOracle  # noqa: E402
from nlss.campaigns.bh_bo import BHBotorchBO, has_botorch  # noqa: E402
from nlss.campaigns.bh_ballet import BHBalletLevelSet  # noqa: E402
from nlss.campaigns.bh_qd import BhQDMAPElites  # noqa: E402
from nlss.campaigns.bh_rankflow import BHRankFlow  # noqa: E402
from nlss.campaigns.bh_tpe import BhTPE  # noqa: E402
from nlss.campaigns.bh_genbo import BHGenBO  # noqa: E402
from nlss.campaigns.bh_rankflow_real import BHRankFlowReal  # noqa: E402
from nlss.campaigns.bh_morgan import morgan_features, morgan_similarity_regions  # noqa: E402
from nlss.campaigns.recovery_eval import RecoveryEvalResult, aurc, evaluate_recovery  # noqa: E402
from nlss.adapters.bh.data import BHData  # noqa: E402

_COMPONENT_KEYS = ("aryl_halide", "ligand", "base", "additive")


def _onehot_features(data: BHData) -> dict[tuple, np.ndarray]:
    """Outcome-blind component one-hot per candidate (same as BHAdapter._siso)."""
    vocab = data.component_vocab()
    out: dict[tuple, np.ndarray] = {}
    for cand, _ in data.records:
        parts: list[float] = []
        for i, key in enumerate(_COMPONENT_KEYS):
            domain = vocab[key]
            hot = [0.0] * len(domain)
            val = cand[i]
            if val in domain:
                hot[domain.index(val)] = 1.0
            parts.extend(hot)
        out[cand] = np.asarray(parts, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Acquisition drivers
# ---------------------------------------------------------------------------


def acquire_random(oracle, queried: set, rng, n: int) -> list:
    avail = [c for c in oracle.candidates if c not in queried]
    rng.shuffle(avail)
    return avail[:n]


def acquire_hillclimb(oracle, queried: set, known: dict, vocab, rng, n: int) -> list:
    pool = set(oracle.candidates)
    if not known:
        return acquire_random(oracle, queried, rng, n)
    best = max(known, key=lambda c: known[c])
    b = list(best)
    proposed = []
    for slot_i, slot in enumerate(_COMPONENT_KEYS):
        for val in vocab[slot]:
            cand = tuple(b[:slot_i] + [val] + b[slot_i + 1:])
            if cand in pool and cand not in queried:
                proposed.append(cand)
    rng.shuffle(proposed)
    return proposed[:n]


def _run_arm(
    oracle: BHFiniteOracle,
    arm: str,
    seed: int,
    cfg: CampaignConfig,
    features: dict[tuple, np.ndarray],
    vocab,
    ledger: ResourceLedger,
    region_components: dict | None = None,
    morgan: dict | None = None,
):
    rng = random.Random(seed)
    pool = list(oracle.candidates)
    n_sol = len(oracle.solution_set)
    queried: set = set()
    known: dict = {}
    prefix_hits: dict[int, int] = {}
    prefix_found_recall: dict[int, float] = {}
    prefix_unseen_recall: dict[int, float] = {}
    prefix_best_yield: dict[int, float] = {}
    checkpoint_metrics: dict[int, dict] = {}
    gp: BHPoolGP | BHGraphProp | None = None
    bo: BHBotorchBO | None = None
    lset: BHBalletLevelSet | None = None
    qd: BhQDMAPElites | None = None
    tpe: BhTPE | None = None
    genbo: BHGenBO | None = None
    rf: BHRankFlowReal | None = None
    surr_seconds: list = []

    def found_recall():
        found = sum(1 for c in queried if oracle.solution_membership(c))
        return found / n_sol if n_sol else 0.0

    def best_yield_so_far():
        return max(known.values()) if known else 0.0

    def eval_at(budget):
        # Recovery over unqueried candidates (posterior-based;).  Only
        # meaningful for arms that produce a genuine posterior (NLSS and, later,
        # GP-BO/DKL-BO).  Bare floor arms (random/hillclimb) have no unqueried
        # posterior, so we report found_recall for them and unseen metrics as NaN.
        unqueried = [c for c in pool if c not in queried]
        if arm.startswith("nlss") or arm == "rankflow_style":
            if gp and gp.trained:
                prob_map = gp.predict_all_solution_prob(unqueried)
            else:
                prob_map = {c: 0.5 for c in unqueried}
            prob_fn = lambda c: prob_map.get(c, 0.5)  # noqa
            supported = [c for c in unqueried if prob_map.get(c, 0.5) >= 0.5]
            res = evaluate_recovery(
                unqueried_candidates=unqueried,
                true_membership=oracle.solution_membership,
                pred_solution_prob=prob_fn,
                true_solution_prob=lambda c: 1.0 if oracle.solution_membership(c) else 0.0,
                recovered_supported=supported,
                node_features=features,
                true_components_of=(lambda c: region_components.get(c)) if region_components else None,
                region_rank_K=60,
            )
            return res
        elif arm == "tpe" and tpe is not None and tpe.trained:
            prob_map = tpe.predict_all_solution_prob(unqueried)
            prob_fn = lambda c: prob_map.get(c, 0.5)
            supported = [c for c in unqueried if prob_map.get(c, 0.5) >= 0.5]
            return evaluate_recovery(
                unqueried_candidates=unqueried,
                true_membership=oracle.solution_membership,
                pred_solution_prob=prob_fn,
                true_solution_prob=lambda c: 1.0 if oracle.solution_membership(c) else 0.0,
                recovered_supported=supported,
                node_features=features,
                true_components_of=(lambda c: region_components.get(c)) if region_components else None,
                region_rank_K=60,
            )
        elif arm in ("ballet_level", "ballet_cov") and lset is not None and lset.trained:
            prob_map = lset.predict_all_solution_prob(unqueried)
            prob_fn = lambda c: prob_map.get(c, 0.5)  # noqa
            supported = [c for c in unqueried if prob_map.get(c, 0.5) >= 0.5]
            return evaluate_recovery(
                unqueried_candidates=unqueried,
                true_membership=oracle.solution_membership,
                pred_solution_prob=prob_fn,
                true_solution_prob=lambda c: 1.0 if oracle.solution_membership(c) else 0.0,
                recovered_supported=supported,
                node_features=features,
                true_components_of=(lambda c: region_components.get(c)) if region_components else None,
                region_rank_K=60,
            )
        elif arm in ("bo_gp", "bo_dkl") and bo is not None and bo.trained:
            # Score the strong-BO surrogate as a *recovery* readout over unseen
            # candidates (P(y>=gamma) from its GP posterior).  This makes the
            # NLSS-vs-BO confrontation on solution-space recovery fair and
            # decisive (report 9.13 / 9.9) instead of NLSS-only.
            prob_map = bo.predict_all_solution_prob(unqueried)
            prob_fn = lambda c: prob_map.get(c, 0.5)  # noqa
            supported = [c for c in unqueried if prob_map.get(c, 0.5) >= 0.5]
            return evaluate_recovery(
                unqueried_candidates=unqueried,
                true_membership=oracle.solution_membership,
                pred_solution_prob=prob_fn,
                true_solution_prob=lambda c: 1.0 if oracle.solution_membership(c) else 0.0,
                recovered_supported=supported,
                node_features=features,
                true_components_of=(lambda c: region_components.get(c)) if region_components else None,
                region_rank_K=60,
            )
        # No posterior: mark unseen-recovery metrics as non-applicable.
        return RecoveryEvalResult(
            n_unqueried=len(unqueried), unseen_recall=float("nan"),
            unseen_precision=float("nan"), unseen_f1=float("nan"),
            pr_auc=float("nan"), region_recall=float("nan"),
            uncovered_distance=None, brier=float("nan"), ece=float("nan"),
        )

    # initial batch
    init = acquire_random(oracle, queried, rng, cfg.n_initial)
    for c in init:
        known[c] = oracle.evaluate(c)
        queried.add(c)
        ledger.bump_oracle()
    # refit for nlss
    if arm == "nlss_rbf":
        gp = BHPoolGP(oracle, features)
        gp.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)
    elif arm == "nlss_graph":
        gp = BHGraphProp(oracle)
        gp.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)
    elif arm == "nlss_hybrid":
        # same structural graph, but readout blends graph logit with global
        # component-marginal log-odds (PR-AUC 0.26->0.36, p<1e-5 at 10 seeds)
        gp = BHGraphProp(oracle, hybrid_alpha=0.6)
        gp.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)
    elif arm == "rankflow_style":
        gp = BHRankFlow(oracle, temp=1.0)
        gp.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)
    elif arm in ("bo_gp", "bo_dkl"):
        bo_feats = morgan if (arm == "bo_dkl" and morgan is not None) else features
        bo = BHBotorchBO(oracle, bo_feats, kernel=("dkl" if arm == "bo_dkl" else "rbf"))
        bo.fit(known)
        ledger.bump_surrogate_fit(0.0)
    elif arm in ("ballet_level", "ballet_cov"):
        lset = BHBalletLevelSet(oracle, features)
        lset.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)
    elif arm == "qd_me":
        qd = BhQDMAPElites(oracle)
        qd.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)
    elif arm == "tpe":
        tpe = BhTPE(oracle)
        tpe.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)
    elif arm == "genbo":
        genbo = BHGenBO(oracle)
        genbo.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)
    elif arm == "rankflow_real":
        rf = BHRankFlowReal(oracle)
        rf.fit(known, surr_seconds)
        ledger.bump_surrogate_fit(surr_seconds[-1] if surr_seconds else 0.0)

    budget_so_far = cfg.n_initial
    prefix_hits[budget_so_far] = sum(1 for c in queried if oracle.solution_membership(c))
    prefix_found_recall[budget_so_far] = found_recall()
    prefix_best_yield[budget_so_far] = best_yield_so_far()
    res0 = eval_at(budget_so_far)
    prefix_unseen_recall[budget_so_far] = res0.unseen_recall
    checkpoint_metrics[budget_so_far] = res0.as_dict()
    checkpoint_metrics[budget_so_far]["found_recall"] = found_recall()
    checkpoint_metrics[budget_so_far]["solutions_found"] = prefix_hits[budget_so_far]
    checkpoint_metrics[budget_so_far]["best_yield"] = best_yield_so_far()

    for rnd in range(cfg.rounds):
        budget_so_far += cfg.batch_size
        if arm == "random":
            batch = acquire_random(oracle, queried, rng, cfg.batch_size)
        elif arm == "hillclimb":
            batch = acquire_hillclimb(oracle, queried, known, vocab, rng, cfg.batch_size)
        elif arm in ("bo_gp", "bo_dkl"):
            batch = bo.acquire(queried, cfg.batch_size)
        elif arm in ("ballet_level", "ballet_cov"):
            cov = 0.5 if arm == "ballet_cov" else 0.0
            batch = lset.acquire(queried, cfg.batch_size, rng, coverage=cov)
            surr_seconds.clear()
            lset.fit(known, surr_seconds)
            if surr_seconds:
                ledger.bump_surrogate_fit(surr_seconds[-1])
        elif arm == "qd_me":
            batch = qd.acquire(queried, cfg.batch_size, rng)
            surr_seconds.clear()
            qd.fit(known, surr_seconds)
            if surr_seconds:
                ledger.bump_surrogate_fit(surr_seconds[-1])
        elif arm == "tpe":
            batch = tpe.acquire(queried, cfg.batch_size, rng)
            surr_seconds.clear()
            tpe.fit(known, surr_seconds)
            if surr_seconds:
                ledger.bump_surrogate_fit(surr_seconds[-1])
        elif arm == "genbo":
            batch = genbo.acquire(queried, cfg.batch_size, rng)
            surr_seconds.clear()
            genbo.fit(known, surr_seconds)
            if surr_seconds:
                ledger.bump_surrogate_fit(surr_seconds[-1])
        elif arm == "rankflow_real":
            batch = rf.acquire(queried, cfg.batch_size, rng)
            surr_seconds.clear()
            rf.fit(known, surr_seconds)
            if surr_seconds:
                ledger.bump_surrogate_fit(surr_seconds[-1])
        elif arm.startswith("nlss"):
            avail = [c for c in pool if c not in queried]
            prob_map = gp.predict_all_solution_prob(avail)
            scored = sorted(avail, key=lambda c: -prob_map.get(c, 0.5))
            batch = scored[: cfg.batch_size]
        elif arm == "rankflow_style":
            batch = gp.acquire(queried, cfg.batch_size, rng)
        else:
            raise ValueError(f"unknown arm {arm}")
        for c in batch:
            known[c] = oracle.evaluate(c)
            queried.add(c)
            ledger.bump_oracle()
        if arm.startswith("nlss") or arm == "rankflow_style":
            surr_seconds.clear()
            gp.fit(known, surr_seconds)
            if surr_seconds:
                ledger.bump_surrogate_fit(surr_seconds[-1])
        elif arm in ("bo_gp", "bo_dkl"):
            import time as _t

            t0 = _t.time()
            bo.fit(known)
            ledger.bump_surrogate_fit(_t.time() - t0)
        prefix_hits[budget_so_far] = sum(1 for c in queried if oracle.solution_membership(c))
        prefix_found_recall[budget_so_far] = found_recall()
        prefix_best_yield[budget_so_far] = best_yield_so_far()
        res = eval_at(budget_so_far)
        prefix_unseen_recall[budget_so_far] = res.unseen_recall
        if budget_so_far in cfg.checkpoints or (rnd + 1) in (2, 4, 6, 8, 10):
            checkpoint_metrics[budget_so_far] = res.as_dict()
            checkpoint_metrics[budget_so_far]["found_recall"] = found_recall()
            checkpoint_metrics[budget_so_far]["solutions_found"] = prefix_hits[budget_so_far]
            checkpoint_metrics[budget_so_far]["best_yield"] = best_yield_so_far()

    a = aurc(prefix_found_recall)
    a_unseen = aurc(prefix_unseen_recall)
    last = checkpoint_metrics.get(max(checkpoint_metrics), res0.as_dict())
    return {
        "prefix_hits": prefix_hits,
        "prefix_found_recall": prefix_found_recall,
        "prefix_unseen_recall": prefix_unseen_recall,
        "prefix_best_yield": prefix_best_yield,
        "checkpoint_metrics": checkpoint_metrics,
        "aurc_found_recall": a,
        "aurc_unseen_recall": a_unseen,
        "n_solutions_found_total": max(prefix_hits.values()) if prefix_hits else 0,
        "best_yield_final": max(prefix_best_yield.values()) if prefix_best_yield else 0.0,
        "solution_prevalence": oracle.solution_prevalence(),
        "final_metrics": last,
        "n_solution_set": len(oracle.solution_set),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="BH recovery campaign runner")
    ap.add_argument("--arms", default="random,nlss_graph,nlss_rbf,hillclimb")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-initial", type=int, default=40)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--out", default="results/bh_campaign")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cfg = CampaignConfig(
        n_initial=args.n_initial,
        batch_size=args.batch,
        rounds=args.rounds,
        checkpoints=(args.n_initial + i * args.batch for i in range(1, args.rounds + 1)),
    )
    # freeze checkpoints to the set if it falls inside the budget
    cps = list(cfg.checkpoints)
    cfg = CampaignConfig(
        n_initial=args.n_initial, batch_size=args.batch, rounds=args.rounds, checkpoints=tuple(cps)
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    commit = git_commit(repo_root)
    cfg_hash = hash_config(
        {
            "benchmark": "bh",
            "n_initial": args.n_initial,
            "batch": args.batch,
            "rounds": args.rounds,
            "arms": arms,
        }
    )

    data = BHData()
    oracle = BHFiniteOracle(data, top_frac=0.05)
    features = _onehot_features(data)
    morgan = morgan_features(oracle)
    vocab = data.component_vocab()
    # Frozen outcome-blind region geometry (): one-factor connected components
    # of the pool, shared across seeds (does not depend on observations).
    # Report 9.5: one-factor c.c. collapses to a single region (RegionRecall
    # degenerates).  Use Morgan k-NN similarity regions as the frozen geometry so
    # RegionRecall is a discriminating recovery endpoint.
    region_components, n_components = morgan_similarity_regions(oracle, k=5)
    n_sol_regions = len({region_components[c] for c in oracle.solution_set})
    print(f"[bh-campaign] pool Morgan-kNN regions={n_components}, solution regions={n_sol_regions}")

    os.makedirs(args.out, exist_ok=True)
    print(f"[bh-campaign] pool={len(oracle.candidates)} |S*|={len(oracle.solution_set)} "
          f"prevalence={oracle.solution_prevalence():.3f} |S*| gamma={oracle.gamma:.2f}")

    per_arm = {arm: [] for arm in arms}
    budget_cps = sorted({b for b in cfg.checkpoints})
    print("\n[bh-campaign] cumulative #S* found (mean over seeds):")
    print("  " + "".join(f"{b:>8}" for b in budget_cps))
    for arm in arms:
        for seed in range(args.seeds):
            run = RunConfig(
                benchmark="bh", method=arm, seed=seed,
                config_hash=cfg_hash, git_commit=commit,
            )
            ledger = ResourceLedger(run)
            result = _run_arm(
                oracle, arm, seed, cfg, features, vocab, ledger, region_components, morgan
            )
            ledger.set(final_metrics=result["final_metrics"],
                       checkpoint_metrics=result["checkpoint_metrics"])
            # write seed log
            log = ledger.to_dict()
            log["final_metrics"].update(
                {
                    "solutions_found_total": result["n_solutions_found_total"],
                    "best_yield": result["best_yield_final"],
                    "solution_prevalence": result["solution_prevalence"],
                }
            )
            fp = os.path.join(args.out, f"bh_{arm}_seed{seed:02d}.json")
            with open(fp, "w") as f:
                json.dump(log, f, indent=2)
            per_arm[arm].append(result)
        # summary row for #S* found
        rows = []
        for b in budget_cps:
            if b <= cfg.total_budget:
                vals = [r["prefix_hits"].get(b, 0) for r in per_arm[arm]]
                rows.append(f"{np.mean(vals):7.1f}±{np.std(vals):.1f}")
        print(f"  {arm:<16}" + "".join(rows))

    # AURC table — primary = found-recall vs budget (comparable across all arms,
    # the feasibility-pilot headroom formulation); NLSS also reports unseen-recall.
    print("\n[bh-campaign] AURC (found-recall = #S* found / |S*|, vs budget):")
    for arm in arms:
        vals = [r["aurc_found_recall"] for r in per_arm[arm]]
        print(f"  {arm:<16} AURC={np.mean(vals):.4f}±{np.std(vals):.4f}")
    print("[bh-campaign] AURC (unseen-recall, posterior over unqueried; NLSS-only):")
    for arm in arms:
        vals = [r["aurc_unseen_recall"] for r in per_arm[arm]]
        if all(np.isnan(v) for v in vals):
            print(f"  {arm:<16} AURC-unseen= n/a (no posterior)")
        else:
            print(f"  {arm:<16} AURC-unseen={np.nanmean(vals):.4f}±{np.nanstd(vals):.4f}")

    print("\n[bh-campaign] final best-yield attained (mean±sd):")
    for arm in arms:
        vals = [r["best_yield_final"] for r in per_arm[arm]]
        print(f"  {arm:<16} BestYield@{cfg.total_budget}={np.mean(vals):.2f}±{np.std(vals):.2f}")

    print("\n[bh-campaign] final recovery (mean±sd, per arm; unseen = posterior arms only):")
    keys = ["found_recall", "unseen_recall", "unseen_precision", "unseen_f1", "pr_auc", "region_recall", "brier"]
    hdr = "  " + "".join(f"{k:>16}" for k in keys)
    print(hdr)
    for arm in arms:
        row = []
        for k in keys:
            vals = []
            for r in per_arm[arm]:
                v = r["final_metrics"].get(k)
                vals.append(v if v is not None else float("nan"))
            label = "n/a" if all(np.isnan(vals)) else f"{np.nanmean(vals):16.4f}"
            row.append(label)
        print(f"  {arm:<16}" + "".join(row))

    print(f"\n[bh-campaign] DONE — logs under {args.out}/")


if __name__ == "__main__":
    main()
