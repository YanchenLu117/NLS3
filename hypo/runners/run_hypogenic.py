#!/usr/bin/env python3
"""HypoGeniC external-relevance runner (C4).

Generates a hypothesis bank on the train split with the official
DefaultGeneration/DefaultUpdate loop, then evaluates held-out accuracy on the
test split. One dataset per invocation:

v1 bug: called nonexistent inference_class.batched_inference and expected
res["label"]; DefaultInference only exposes batched_predict/run_inference_final
which return (predictions, actual_labels). Generation phase was fine (38 min
for 100 hypotheses) — only eval crashed with AttributeError. v2:
  1. eval via run_inference_final(test_data, bank, generate_kwargs=...)
  2. bank saved to out_folder/bank_final.json right after generation
  3. single-dataset mode (argv[1]) so external `timeout` bounds one attempt
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repo root
HB = REPO / "hypo" / "data"
OUT = REPO / "work" / "external_baselines"
OUT.mkdir(parents=True, exist_ok=True)
# auth: any OpenAI-compatible endpoint. OPENAI_API_KEY is required by the
# openai client; base URL may come from NLSS_LLM_BASE_URL (paper convention)
# or OPENAI_BASE_URL (openai convention).
assert os.environ.get("NLSS_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"), (
    "set NLSS_LLM_BASE_URL (and NLSS_LLM_MODEL) to your OpenAI-compatible endpoint")
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("NLSS_LLM_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("NLSS_LLM_BASE_URL", ""))


def run_one(ds: str, model_name: str | None = None, num_train: int = 75,
            num_init: int = 10) -> dict:
    import random
    import numpy as np
    model_name = model_name or os.environ.get("NLSS_LLM_MODEL", "")
    sys.path.insert(0, str(REPO / "hypo"))
    os.chdir(REPO / "hypo")
    from hypogenic.extract_label import extract_label_register
    from hypogenic.tasks import BaseTask
    from hypogenic.prompt import BasePrompt
    from hypogenic.utils import set_seed
    from hypogenic.algorithm.generation import DefaultGeneration
    from hypogenic.algorithm.inference import DefaultInference
    from hypogenic.algorithm.replace import DefaultReplace
    from hypogenic.algorithm.update import DefaultUpdate
    from hypogenic.LLM_wrapper import llm_wrapper_register
    from hypogenic.logger_config import LoggerConfig
    LoggerConfig.setup_logger(level="WARNING")

    t0 = time.time()
    cfg_path = HB / ds / "config.yaml"
    api = llm_wrapper_register.build("gpt")(model=model_name, path_name=None)
    task = BaseTask(str(cfg_path), extract_label=None, from_register=extract_label_register)
    seed = 42
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    train_data, _, _ = task.get_data(num_train, 25, 25, seed)
    prompt_class = BasePrompt(task)
    inference_class = DefaultInference(api, prompt_class, train_data, task)
    generation_class = DefaultGeneration(api, prompt_class, inference_class, task)
    out_folder = OUT / ds / model_name
    out_folder.mkdir(parents=True, exist_ok=True)
    update_class = DefaultUpdate(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=DefaultReplace(20),
        save_path=str(out_folder),
        num_init=num_init,
        k=10,
        alpha=5e-1,
        update_batch_size=10,
        num_hypotheses_to_update=1,
        save_every_n_examples=10,
    )
    bank = update_class.batched_initialize_hypotheses(
        num_init, init_batch_size=10, init_hypotheses_per_batch=10,
        cache_seed=None, temperature=1e-5, max_tokens=1000,
    )
    n_gen = len(bank)
    # persist bank so a future eval-only pass can recover without regeneration
    try:
        (out_folder / "bank_final.json").write_text(json.dumps(
            {h: {"acc": float(getattr(bank[h], "acc", 0.0)),
                 "reward": float(getattr(bank[h], "reward", 0.0))}
             for h in bank}, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        print(f"BANK_SAVE_WARN:{type(e).__name__}", flush=True)

    gen_s = round(time.time() - t0, 1)
    # official eval: run_inference_final -> (predictions, actual_labels)
    # FIX (DEV-20260918-hypogenic-eval-unpack): get_data returns
    # (train, test, val); the old unpack grabbed the empty train slot,
    # so run_inference_final got 0 test samples and acc was silently None.
    _, test_data, _ = task.get_data(0, 50, 0, seed)
    acc = None
    try:
        preds, gold = inference_class.run_inference_final(
            test_data, bank, cache_seed=None,
            generate_kwargs={"temperature": 1e-5, "max_tokens": 1000},
        )
        if preds:
            match = sum(1 for p, g in zip(preds, gold)
                        if str(p).strip().lower() == str(g).strip().lower())
            acc = match / len(gold)
    except Exception as e:
        acc = f"INFER_ERR:{type(e).__name__}"
    rec = {"dataset": ds, "model": model_name, "n_hypotheses": n_gen,
           "gen_s": gen_s, "heldout_accuracy": acc,
           "wall_s": round(time.time() - t0, 1),
           "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "runner": "nls3_public_v1"}
    (OUT / "hypogenic_results_20260917.jsonl").open("a", encoding="utf-8").write(
        json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "deceptive_reviews"
    run_one(ds)
