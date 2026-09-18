# NLS³: Constructing Natural Language Scientific Solution Space

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-green.svg" alt="Python 3.10+">
  <a href="#verify-your-install-2-min-no-llm"><img src="https://img.shields.io/badge/Smoke%20test-passing-brightgreen.svg" alt="Smoke test"></a>
</p>

<p align="center"><i>The NLS³ Authors (Anonymous)</i></p>

---

## Why this matters

Scientific exploration usually starts from a search space someone already
defined — variables chosen, encodings fixed, representations frozen. But when
the relevant scientific variation is itself still open, **how that variation
acquires computational existence is a first-class part of discovery.**

NLS³ makes the construction of the hypothesis space an explicit, auditable
operation, closing a loop:

```
              ┌─────────────────────────────────────────────┐
              │                                             │
              ▼                                             │
  Natural-language hypotheses ──Construct──► Solution space │
                                              │             │
                                              ▼             │
                                     Explore & audit        │
                                              │             │
                                              └── Lift ─────┘
```

**We prescribe verification, not representation.** The hypothesis space is
*constructed*, not given. Natural language gives the proposer freedom to
suggest *what could vary*; a hybrid audit — the LLM proposes, but only
executable, verified structure counts — keeps that freedom honest.

---

## Three runnable components

Each directory is self-contained and independently runnable:

| | Directory | What it demonstrates | Quick command |
|---|---|---|---|
| 🧪 | [`hypo/`](hypo/) | Hypothesis banks built from natural language transfer to four real benchmark datasets | `python3 hypo/runners/run_hypogenic.py headline_binary` |
| 🧬 | [`scripts/`](scripts/) + [`nlss/`](nlss/) | Constructed solution spaces beat fixed-space specialists on two real optimization problems (chemical reaction yield, protein fitness) | `python3 scripts/run_bh_campaign.py --arms random --seeds 1 --out work/bh_smoke` |
| 🔍 | [`audit/`](audit/) | The hybrid-audit engine: schema, proposer, auditor, and open-form benchmark driver | `bash smoke_test.sh` |

## Setup

```bash
pip install -r requirements.txt
```

LLM-backed experiments need any OpenAI-compatible endpoint:

```bash
export NLSS_LLM_BASE_URL="https://your-endpoint/v1"
export NLSS_LLM_MODEL="your-model"
export NLSS_LLM_API_KEY="your-key"        # if your endpoint requires one
```

Baselines that never call an LLM (random, hill-climb, GP/BO family) run with
**no configuration at all**.

## Verify your install (≈2 min, no LLM)

```bash
bash smoke_test.sh
```

Runs both optimization baselines plus import checks for every component.
Expected final line: **`SMOKE_OK`**.

---

## 🧪 1. External relevance — hypothesis generation evaluated on real data

The `hypo/` component generates a hypothesis bank on a dataset's train split
with the official HypoGeniC generation loop, then measures accuracy on the
held-out test split. Four MIT-licensed datasets are included
(see [`hypo/data/`](hypo/data)):

```bash
python3 hypo/runners/run_hypogenic.py headline_binary
# also: deceptive_reviews · dreaddit · persuasive_pairs
```

Held-out accuracy reported in the paper:

| deceptive_reviews | headline_binary | persuasive_pairs | dreaddit |
|:---:|:---:|:---:|:---:|
| 0.38 | 0.54 | 0.80 | 0.70 |

## 🧬 2. Downstream optimization — constructed spaces vs fixed spaces

The optimization suites compare substrates *constructed* by NLS³ against
strong fixed-space specialists on two real problems:

- **Buchwald–Hartwig coupling** (chemistry): predict-and-optimize reaction yield
- **GB1 protein** (biology): maximize measured fitness over a combinatorial variant pool

```bash
# no-LLM sanity check (runs in seconds):
python3 scripts/run_gb1_campaign.py --arms random --seeds 1 --out work/gb1_smoke

# compare a specialist against the random-policy floor:
python3 scripts/run_gb1_campaign.py --arms gb1_gpbo,random --seeds 5 --out work/gb1_5seed
```

Available arms:

| Benchmark | Constructed substrate | Fixed-space specialists | Policy baselines |
|---|---|---|---|
| BH | `nlss_graph` | `bo_gp` · `ballet_level` · `bo_dkl` · `gryffin_plain` | `random` · `hillclimb` |
| GB1 | `nlss_hamming` | `gb1_gpbo` · `gb1_ballet` · `gb1_alde` | `random` · `hillclimb` |

The full preregistered protocol — frozen experiment registry, Holm-corrected
confirmatory family, equivalence gates, all arms × seeds — is
[`scripts/run_preregistered_protocol.py`](scripts/run_preregistered_protocol.py) (see `--help`).

**Data.** BH ships with the repository
([`data/bh/`](data/bh), MIT, from the Doyle lab). The full 149k-variant GB1
fitness table is not redistributed; fetch it with

```bash
bash scripts/download_gb1_data.sh
```

Without it, the GB1 loader transparently falls back to a small built-in
measured subset so that every arm remains runnable.

## 🔍 3. The hybrid-audit engine

[`audit/nlss_v10/`](audit/nlss_v10) is the engine behind the Construct →
Explore → Evolve loop:

```
audit/nlss_v10/
├── schema/       substrate schema + executable-operator semantics
├── construct/    hypothesis proposal with content-addressed prompt provenance
├── audit/        the hybrid auditor (score dimensions, hard obligations, judge client)
├── explore/      fast/slow feedback extraction
├── metrics/      arm-level summary statistics
└── bench/        open-form benchmark driver (open vs menu-constrained arms)
```

A substrate is a *declared, executable* hypothesis space: the proposer
describes what could vary in natural language, the schema forces that
description into runnable structure, and the auditor scores how well the
declared semantics survive execution. Every prompt carries a hash of its own
rendered content, so provenance is checkable after the fact.

```python
import sys; sys.path.insert(0, "audit")
from nlss_v10.schema.chi_schema import validate
from nlss_v10.audit.engine import HybridAuditor, SCORE_DIMS

result = validate(chi_dict)          # structural + semantic checks
auditor = HybridAuditor(judge_client)  # wire your LLM into audit/judge_client.py
```

---

## Repository layout

```
NLS3/
├── hypo/                  HypoGeniC package (vendored, MIT) + datasets + runner
│   ├── hypogenic/         upstream hypothesis-generation library
│   ├── data/              four benchmark datasets (MIT, from HypoBench)
│   └── runners/           generate-and-evaluate entry point
├── nlss/                  shared optimization-suite package
│   ├── campaigns/         per-benchmark search strategies
│   ├── adapters/          dataset loaders (BH, GB1)
│   ├── stats/             Holm correction, equivalence gates
│   └── prereg/            frozen experiment registry
├── scripts/               campaign entry points + full preregistered protocol
├── audit/nlss_v10/        hybrid-audit engine (see above)
├── data/bh/               Buchwald–Hartwig reaction data (MIT, shipped)
├── smoke_test.sh          2-minute verification of all three components
├── THIRD_PARTY.md         licenses and provenance for every vendored component
└── requirements.txt
```

## What is intentionally not here

- Experiment outputs (raw runs, result tables, cost ledgers, preregistration
  snapshots) — the paper's numbers are reproducible from the code above, not
  shipped as artifacts
- Endpoints, credentials, or model names — configure your own via env vars

## License

Our code is released under **MIT** (see [`LICENSE`](LICENSE)). Vendored
third-party components keep their upstream licenses — HypoGeniC (MIT),
HypoBench datasets (MIT), BH reaction data (MIT); the GB1 fitness table is
download-on-demand with a built-in measured fallback. Details in
[`THIRD_PARTY.md`](THIRD_PARTY.md).
