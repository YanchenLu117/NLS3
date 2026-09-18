"""V7 Core — unified artifact pipeline (EXPERIMENT_PLAN §12.2).

RunArchiver writes every standardized artifact per round into a run directory so
each run/round is clean, accurate, replayable, and auditable (it is the carrier
for the plan's "official + NLSS standardized outputs").  It consumes only the
public NLSSModel surface; it never mutates the model.  The exploration gear is
recorded DYNAMICALLY (read from the model), never hardcoded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.contract import contract_diff

# ---- JSON helpers (handles numpy scalar) ----------------------------------


def _np(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return _np(obj)


def _fingerprint(record) -> str:
    """Stable fingerprint for one evidence record (lexer idempotency)."""
    try:
        payload = getattr(record, "metadata", None) or {}
        h = record.hypothesis
        o = record.object
        obs = record.observation
        parts = [
            record.kind,
            h.hypothesis_id if h else "",
            o.object_id if o else "",
            obs.object_id if obs else "",
            str(obs.value if obs else ""),
        ]
        return "|".join(parts)
    except Exception:
        return str(record)


# ---- RunArchiver -----------------------------------------------------------

class RunArchiver:
    """Writes the §12.2 per-round artifact layout into ``run_dir``."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        commit: str | None = None,
        env_lock: str | None = None,
        dataset_hashes: Mapping[str, str] | None = None,
        model_id: str | None = None,
        seed: int | None = None,
        regime: str = "controlled",
    ) -> None:
        self.root = Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_meta = {
            "commit": commit,
            "env_lock": env_lock,
            "dataset_hashes": dict(dataset_hashes or {}),
            "model_id": model_id,
            "seed": seed,
            "regime": regime,
            "started_at": time.time(),
        }
        self._manifest_written = False
        self._recorded: set[str] = set()
        self._last_spec = None
        # seed idempotency from an existing ledger (cross-instance re-archive)
        ledger = self.root / "evidence" / "ledger.jsonl"
        if ledger.exists():
            try:
                for line in ledger.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if isinstance(rec, dict) and rec.get("fingerprint"):
                        self._recorded.add(rec["fingerprint"])
            except Exception:
                pass

    def __enter__(self) -> "RunArchiver":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    # -- low-level writers ---------------------------------------------------

    def write_jsonl(self, rel: str, data: Any) -> Path:
        """Public line-oriented JSONL writer (one JSON object per line)."""
        return self._append_jsonl(rel, data)

    def _write(self, rel: str, data: Any) -> Path:
        if rel.endswith(".jsonl"):
            return self._append_jsonl(rel, data)  # line-oriented JSON, not repr
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith((".txt", ".md")):
            text = "\n".join(str(_json_safe(x)) for x in (data if isinstance(data, list) else [data]))
            text = text if rel.endswith((".jsonl", ".txt")) or isinstance(data, str) else str(data)
            p.write_text(text + "\n")
        else:
            p.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2, default=str))
        return p

    def _append_jsonl(self, rel: str, data: Any) -> Path:
        """Append one JSON object per line (line-oriented .jsonl)."""
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        items = data if isinstance(data, (list, tuple)) else [data]
        with p.open("a", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(_json_safe(it), ensure_ascii=False, default=str) + "\n")
        return p

    def _write_manifest(self) -> Path:
        if not self._manifest_written:
            p = self.root / "run_manifest.json"
            if p.exists():
                try:
                    existing = json.loads(p.read_text())
                    existing.update({k: v for k, v in self.manifest_meta.items() if v is not None})
                    p.write_text(json.dumps(existing, ensure_ascii=False, indent=2, default=str))
                    self._manifest_written = True
                    return p
                except Exception:
                    pass
            self._manifest_written = True
            return self._write("run_manifest.json", self.manifest_meta)
        return self.root / "run_manifest.json"

    # -- collect from the model surface --------------------------------------

    def _posterior_table(self, model) -> list[dict]:
        out: list[dict] = []
        state = model.state
        if state.posterior is None:
            return out
        for oid in state.objects:
            try:
                m = state.posterior.marginal(oid)
                out.append({
                    "object_id": oid,
                    "mean": _np(m.mean),
                    "std": _np(m.std),
                    "solution_probability": _np(m.solution_probability),
                })
            except Exception:
                continue
        return out

    def _calibration(self, model) -> dict[str, Any]:
        state = model.state
        if state.posterior is None:
            return {"note": "no posterior", "ece": None}
        pairs = []
        for oid, obs in state.observations.items():
            try:
                m = state.posterior.marginal(oid)
                if m.solution_probability is not None:
                    pairs.append((float(m.solution_probability), float(obs.value)))
            except Exception:
                continue
        if not pairs:
            return {"note": "no paired posterior/observation", "ece": None}
        probs = np.array([p for p, _ in pairs])
        labels = np.array([1.0 if v > 0 else 0.0 for _, v in pairs])
        bin_edges = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            m = (probs > lo) & (probs <= hi)
            if m.any():
                ece += (m.sum() / len(probs)) * abs(probs[m].mean() - labels[m].mean())
        return {"note": "bin-ECE (solution_probability vs positive observation)", "ece": _np(ece), "n": len(pairs)}

    def _controller_decision(self, model, controller_decision) -> dict[str, Any]:
        """§12.2 decision record: gear, channel, channel probability, candidate
        mask, selection probability, and fallback — derived from the controller
        surface when no caller-supplied decision is given (Attack D/E audit)."""
        if controller_decision is not None:
            return dict(controller_decision)
        gear = model.exploration
        ctl = model.exploration_controller
        profile = ctl.profile(gear)
        import random

        channel = ctl.select_channel(random.Random(0), gear)
        w = profile.weight_of(channel)
        object_ids = list(model.state.objects)
        pol = {}
        if channel != "OPEN" and object_ids and model.state.posterior is not None:
            try:
                pol = ctl.normalized(channel, object_ids, model.state.posterior, model.eta)
            except Exception:
                pol = {}
        top_id = max(pol, key=pol.get) if pol else None
        return {
            "gear": gear.value,  # DYNAMIC — never hardcoded
            "channel": channel,
            "channel_probability": w,
            "candidate_mask": object_ids,
            "selection_probability": round(pol[top_id], 6) if top_id else None,
            "selection_candidate": top_id,
            "fallback": "default-derived-from-controller",
            "w": list(profile.weights),
        }

    def _write_figure(self, model, *, round_idx: int) -> Path:
        """Best-effort solution-space figure (§12.2 #17).  Uses matplotlib when
        available; otherwise falls back to a JSON summary so the artifact slot
        always exists (the plan's full §13 figures are a later step)."""
        t = round_idx
        slot = self.root / "figures"
        slot.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return self._write(f"figures/solution_space_round_{t}.json",
                               {"note": "matplotlib unavailable; no pdf", "round": t})
        try:
            state = model.state
            fig, ax = plt.subplots(figsize=(4, 3))
            pts = []
            for oid in state.objects:
                try:
                    m_ = state.posterior.marginal(oid) if state.posterior is not None else None
                    sol = float(m_.solution_probability) if (m_ and m_.solution_probability is not None) else 0.0
                    pts.append((oid, sol))
                except Exception:
                    continue
            if pts:
                xs = list(range(len(pts)))
                ys = [p[1] for p in pts]
                ax.scatter(xs, ys, s=18)
                ax.set_xlabel("object idx"); ax.set_ylabel("p_sol")
            else:
                ax.text(0.5, 0.5, "no posterior", ha="center")
            ax.set_title(f"round {t} — solution space")
            out = slot / f"solution_space_round_{t}.pdf"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            return out
        except Exception:
            return self._write(f"figures/solution_space_round_{t}.json",
                               {"note": "figure render failed", "round": t})

    # -- public entry ---------------------------------------------------------

    def write_round(
        self,
        model,
        *,
        round_idx: int,
        proposals: Sequence[Any] | None = None,
        controller_decision: Mapping[str, Any] | None = None,
        token_budget: int = 1600,
        force: bool = False,
    ) -> dict[str, str]:
        """Write all §12.2 artifacts for one round; returns rel-path -> abs-path."""
        self.manifest_meta.setdefault("token_budget", token_budget)
        if "prompt_hash" not in self.manifest_meta:
            self.manifest_meta["prompt_hash"] = None
        self._write_manifest()
        t = round_idx
        state = model.state
        written: dict[str, str] = {}

        # evidence ledger.jsonl — append unique fingerprints (idempotent re-write)
        for record in model.evidence_log:
            fp = _fingerprint(record)
            if fp in self._recorded and not force:
                continue
            self._recorded.add(fp)
            self._append_jsonl("evidence/ledger.jsonl", {
                "kind": record.kind,
                "fingerprint": fp,
            })

        # representation
        n = len(list((self.root / "representation").glob("contract_v*.json"))) if (self.root / "representation").exists() else 0
        version = n + 1
        spec = model.representation
        written["contract"] = str(self._write(f"representation/contract_v{version}.json", {
            "spec_id": spec.spec_id,
            "source": spec.source,
            "admitted": spec.admitted,
            "typed_variables": [{"name": v.name, "vtype": v.vtype, "role": v.role} for v in spec.typed_variables],
            "relations": [{"name": r.name, "scope": r.scope} for r in spec.relations],
            "transforms": [{"name": tr.name, "executable": tr.executable} for tr in spec.transforms],
            "entities": list(spec.entities),
            "geometry": spec.geometry,
            "grounding": spec.grounding,
            "backend_hint": spec.backend_hint,
            "provenance": spec.provenance,
        }))
        written["schema_diff"] = str(self._write(
            f"representation/schema_diff_v{version-1}_v{version}.json",
            contract_diff(self._last_spec, spec, version=version),
        ))
        self._last_spec = spec
        written["logic_report"] = str(self._write(f"representation/logic_report_v{version}.json", {
            "logic_valid": state.metadata.get("representation_logic_valid"),
            "gate": state.metadata.get("representation_logic_gate"),
        }))
        written["fidelity_report"] = str(self._write(f"representation/fidelity_report_v{version}.json", {
            "certificate": state.metadata.get("representation_fidelity_cert"),
            "fidelity_score": state.metadata.get("representation_fidelity"),
        }))

        # grounding gamma
        gamma_lines = []
        for oid, obj in state.objects.items():
            gamma_lines.append({
                "object_id": oid,
                "canonical_form": obj.canonical_form,
                "display": obj.display_text,
                "observed": oid in state.observations,
            })
        written["gamma"] = str(self._append_jsonl(f"grounding/gamma_round_{t}.jsonl", gamma_lines))

        # field posterior + calibration
        post = self._posterior_table(model)
        npz = self.root / "field"
        npz.mkdir(parents=True, exist_ok=True)
        arr_path = self.root / f"field/posterior_round_{t}.npz"
        if post:
            np.savez(arr_path, object_ids=np.array([r["object_id"] for r in post]),
                     mean=np.array([r["mean"] for r in post]),
                     std=np.array([r["std"] for r in post]),
                     solution_probability=np.array([r["solution_probability"] for r in post]))
        else:
            np.savez(arr_path, object_ids=np.array([]), mean=np.array([]), std=np.array([]), solution_probability=np.array([]))
        written["field_npz"] = str(arr_path)
        written["calibration"] = str(self._write(f"field/calibration_round_{t}.json", self._calibration(model)))

        # readout nl + computable
        nl = model.readout(state, token_budget=token_budget) if hasattr(model, "readout") else ""
        written["readout_nl"] = str(self._write(f"readout/natural_language_round_{t}.md", nl))
        computable = {
            "representational_state": model.representational_state(),
            "solution_set_eta05": list(model.solution_set(state, eta=0.5)) if hasattr(model, "solution_set") else [],
            "n_objects": len(state.objects),
            "n_observations": len(state.observations),
        }
        written["readout_computable"] = str(self._write(f"readout/computable_round_{t}.json", computable))

        # controller decision
        written["controller"] = str(self._write(f"controller/decision_round_{t}.json",
                                                self._controller_decision(model, controller_decision)))

        # proposals + oracle
        written["proposals"] = str(self._append_jsonl(f"proposals/candidates_round_{t}.jsonl",
                                                      list(proposals or [])))
        obs_json = [{"object_id": oid, "value": _np(o.value), "round": _np(o.round_index)}
                    for oid, o in state.observations.items()]
        written["oracle"] = str(self._append_jsonl(f"oracle/observations_round_{t}.jsonl", obs_json))

        # metrics + resources
        written["metrics"] = str(self._write(f"metrics/round_{t}.json", {
            "n_objects": len(state.objects),
            "n_observations": len(state.observations),
            "n_hypotheses": len(state.hypotheses),
            "posterior_fitted": state.posterior is not None,
            "representation_admitted": state.metadata.get("representation_admitted"),
            "fidelity": state.metadata.get("representation_fidelity"),
            "evidence_count": len(model.evidence_log),
        }))
        written["resources"] = str(self._write(f"resources/usage_round_{t}.json", {
            "token_usage": dict(model.token_usage()) if hasattr(model, "token_usage") else {},
            "wall_time": time.time() - float(self.manifest_meta.get("started_at", time.time())),
        }))
        written["figure"] = str(self._write_figure(model, round_idx=t))

        return written

    # -- convenience -----------------------------------------------------------

    def list_artifacts(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in sorted(self.root.rglob("*")) if p.is_file())
