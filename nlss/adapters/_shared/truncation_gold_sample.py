"""E4-0.9 截断金样本 runner（预注册 tripwire: finish_reason != "stop" > 5% → 该系统 native-only + DECISION）。

输入: JSONL, 每行 {"system": <id>, "tag": <prompt 类别>, "text": <真实提示词>}
      —— 提取器(各系统 prompt 构建路径)产出的 20 条真实提示, 预注册于
      work/G_generalists/truncation_gold/{system}_gold_prompts.jsonl。
输出: work/G_generalists/truncation_gold/{system}_gold_result.json
      {rate_finish_ne_stop, n, histogram{finish_reason: count}, rows:[...]}

并发红线: GLM 全局并发上限 10; E 线批在跑时 ≤3 或等间隙(2026-08-31 放行规则)。
本 runner 顺序发送(并发 1), 由调用方协调。

hardened (2026-09-01: openai SDK hangs without explicit timeout when upstream
连接 ESTABLISHED 但零字节回流, poll 永久阻塞):
  - 显式 timeout=600s / max_retries=1 (SDK 内部), 外层重试 5 次 + 退避;
  - 每样本 flush 进度日志;
  - 增量 progress sidecar (<prompts>.progress.jsonl), 重启按 (idx, sha) 续跑,
    判定逻辑与行 schema 不变。

Usage:
    PYTHONPATH=src python -m nlss.adapters._shared.truncation_gold_sample \
        --prompts work/G_generalists/truncation_gold/s2_gold_prompts.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

DEFAULT_MODEL = os.environ.get("NLSS_LLM_MODEL", "")

REQ_TIMEOUT_S = 600.0
REQ_RETRIES = 5


def load_prompts(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load_progress(progress_path: str | None) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if not progress_path or not os.path.exists(progress_path):
        return done
    with open(progress_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done[int(r["idx"])] = r
                except Exception:  # noqa: BLE001
                    continue
    return done


def run_sample(rows: list[dict], model: str, max_tokens: int = 32768,
               progress_path: str | None = None) -> dict:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["NLSS_LLM_KEY"],
        base_url=os.environ.get("NLSS_GLM_BASE_URL"),
        timeout=REQ_TIMEOUT_S,
        max_retries=1,
    )
    done = _load_progress(progress_path)
    out_rows: list[dict] = []
    for i, row in enumerate(rows):
        text = row["text"]
        sha = _sha16(text)
        prev = done.get(i)
        if prev and prev.get("prompt_sha256_16") == sha and (
                prev.get("finish_reason") or prev.get("error")):
            print(f"[{i + 1}/{len(rows)}] resumed idx={i} finish={prev.get('finish_reason')}", flush=True)
            out_rows.append(prev)
            continue
        finish, usage, err = None, None, None
        for attempt in range(REQ_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=max_tokens,
                )
                finish = resp.choices[0].finish_reason
                u = resp.usage
                usage = {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "completion_tokens_details": getattr(u, "completion_tokens_details", None)
                    and str(getattr(u, "completion_tokens_details")),
                }
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                print(f"[{i + 1}/{len(rows)}] attempt={attempt + 1} error: {err[:200]}", flush=True)
                time.sleep(10 * (attempt + 1))
        outcome = finish if finish else f"error:{err}"
        out_rows.append({
            "idx": i, "system": row.get("system"), "tag": row.get("tag"),
            "prompt_sha256_16": sha,
            "prompt_chars": len(text),
            "finish_reason": finish, "usage": usage, "error": err,
        })
        print(f"[{i + 1}/{len(rows)}] tag={row.get('tag')} -> {outcome}", flush=True)
        if progress_path:
            with open(progress_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out_rows[-1], ensure_ascii=False) + "\n")
        time.sleep(1)  # 顺序 + 间隔, 并发红线
    n = len(out_rows)
    # 三分法 (DECISION_E4_GOLD_TAXONOMY_20260831): stop/length/error
    n_stop = sum(1 for r in out_rows if r["finish_reason"] == "stop")
    n_length = sum(1 for r in out_rows if r["finish_reason"] == "length")
    n_error = n - n_stop - n_length
    completed = n_stop + n_length
    # 截断率 = length/(stop+length), 只算完成样本
    rate = round(n_length / completed, 4) if completed else None
    valid = completed >= 18
    histogram = Counter(
        (r["finish_reason"] if r["finish_reason"] else f"error:{r['error']}") for r in out_rows)
    return {
        "model": model,
        "max_tokens": max_tokens,
        "n": n,
        "n_stop": n_stop,
        "n_length": n_length,
        "n_error": n_error,
        "completed": completed,
        "validity": "valid" if valid else "infra-invalid (<18/20 completed)",
        "rate_truncation": rate,
        "legacy_rate_finish_ne_stop": round((n_length + n_error) / n, 4) if n else None,
        "tripwire": ">5% of completed -> system native-model-only + DECISION",
        "histogram": dict(histogram),
        "rows": out_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_tokens", type=int, default=32768,
                    help="GLM cfg value; endpoint accepts 16384/32768 (coordinator-tested, max_model_len=200000)")
    args = ap.parse_args()

    rows = load_prompts(args.prompts)
    pdir = os.path.dirname(os.path.abspath(args.prompts))
    pstem = os.path.splitext(os.path.basename(args.prompts))[0]
    progress = os.path.join(pdir, f"{pstem}.progress.jsonl")
    result = run_sample(rows, args.model, args.max_tokens, progress_path=progress)
    system = rows[0].get("system", "unknown") if rows else "unknown"
    out = args.out or (
        f"work/G_generalists/truncation_gold/{system}_gold_result_{args.max_tokens}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in (
        "model", "n", "n_stop", "n_length", "n_error", "validity",
        "rate_truncation")}, indent=2))
    if not result["validity"].startswith("valid"):
        return 2  # infra-invalid: 不作判定
    return 1 if (result["rate_truncation"] or 0) > 0.05 else 0


if __name__ == "__main__":
    raise SystemExit(main())
