"""judge_client.py — v10 semantic judge client (OpenAI-compatible endpoint).

Role separation (Methods C3): the judge never sees the proposer's chain of
thought; different session; sanitizer strips proposer reasoning before any
diagnostic is returned to the proposer (see audit.sanitize).

Judge scoring protocol:
  - each probe = one chat completion with a frozen rubric prompt;
  - judge returns strict JSON: {"score": 0..1, "confidence": 0..1, "reason": "..."};
  - score 0.0 = definite fail, 1.0 = definite pass; fractional allowed;
  - uncertainty u = combination of judge self-reported confidence radius and
    cross-judge disagreement (second judge optional, v9 judge_sens protocol);
  - conservative score q̂-u is what floors compare against (frozen in prereg).

Endpoint comes from NLSS_LLM_BASE_URL / NLSS_LLM_JUDGE_MODEL / NLSS_LLM_API_KEY env vars.
enable_thinking=false appended by the v9 gateway for qwen models.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

JUDGE_URL = os.environ.get("NLSS_LLM_BASE_URL", "")
JUDGE_MODEL = os.environ.get("NLSS_LLM_JUDGE_MODEL", os.environ.get("NLSS_LLM_MODEL", ""))
API_KEY = os.environ.get("NLSS_LLM_API_KEY", "")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeClient:
    def __init__(self, base_url: str = JUDGE_URL, model: str = JUDGE_MODEL,
                 api_key: str = API_KEY, timeout: float = 480.0,
                 max_retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # cumulative real token usage (cost_monitor reads the delta per audit)
        self.usage = {"in": 0, "out": 0, "n": 0}

    def _post(self, messages: list, temperature: float = 0.0) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "enable_thinking": False,   # gateway/proxy forwards; qwen3 mute
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        last_err = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    obj = json.loads(r.read().decode("utf-8"))
                u = obj.get("usage") or {}
                self.usage["in"] += int(u.get("prompt_tokens") or 0)
                self.usage["out"] += int(u.get("completion_tokens") or 0)
                self.usage["n"] += 1
                return obj["choices"][0]["message"]["content"]
            except Exception as e:  # transient 502/timeout backoff (v9 glmretry protocol)
                last_err = e
                time.sleep(min(60.0, 2 ** attempt * 2))
        raise RuntimeError(f"judge post failed after {self.max_retries} retries: {last_err}")

    def score(self, rubric: str, candidate_payload: str,
              second_judge: bool = False) -> dict:
        """One rubric-scored probe. Returns score/confidence/reason (+ u parts)."""
        prompt = (f"{rubric}\n\n=== CANDIDATE UNDER AUDIT ===\n{candidate_payload}\n\n"
                  "Respond with ONLY a JSON object: "
                  '{"score": <0..1>, "confidence": <0..1>, "reason": "<=120 words"}')
        content = self._post([{"role": "user", "content": prompt}])
        m = _JSON_RE.search(content)
        if not m:
            return {"score": None, "confidence": None, "reason": "unparseable judge output",
                    "raw": content[:500], "parse_fail": True}
        try:
            obj = json.loads(m.group(0))
            obj["score"] = max(0.0, min(1.0, float(obj["score"])))
            obj["confidence"] = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
        except Exception:
            return {"score": None, "confidence": None, "reason": "bad judge JSON",
                    "raw": content[:500], "parse_fail": True}
        u_self = 1.0 - obj["confidence"]
        if second_judge:
            content2 = self._post([{"role": "user", "content": prompt}])
            m2 = _JSON_RE.search(content2)
            try:
                o2 = json.loads(m2.group(0)) if m2 else {}
                s2 = max(0.0, min(1.0, float(o2.get("score", obj["score"]))))
            except Exception:
                s2 = obj["score"]
            obj["score_judge2"] = s2
            obj["u_disagreement"] = abs(obj["score"] - s2) / 2.0
        obj["u"] = u_self + obj.get("u_disagreement", 0.0)
        return obj


def score_pair_agreement(a: dict, b: dict) -> float:
    """Cross-judge disagreement radius for u calibration checks (dev-set only)."""
    if a.get("score") is None or b.get("score") is None:
        return 1.0
    return abs(a["score"] - b["score"]) / 2.0
