# NLS³: Constructing Natural Language Scientific Solution Space

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-green.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Code-Reproducible-brightgreen.svg" alt="Reproducible">
</p>

<p align="center">
  <b>The NLS³ Authors</b> (Anonymous)
</p>

---

Scientific exploration need not always begin from a computational search space
that has already been specified. When the relevant scientific variation is
still open, **deciding how that variation acquires computational existence is a
first-class part of the discovery process.**

NLS³ studies this upstream problem through a three-operation loop:

```
H^{NL}  ──Construct──►  S  ──Explore──►  R_t  ──Lift──►  S_t^{NL}  ──►  …
```

**We prescribe verification, not representation.** The hypothesis space is
constructed (not given); natural language provides proposal freedom; a
*Hybrid Audit* (LLM proposes, verification activates) keeps the constructed
space honest.

This repository contains the three code components behind the paper's
experiments, each independently runnable:

| Component | Paper claim | What it shows |
|---|---|---|
| [`hypo/`](#1-external-relevance-hypogenic-c4) | **C4** — external relevance | HypoGeniC hypothesis banks transfer to four real benchmark datasets |
| [`nlss/` + `scripts/`](#2-downstream-optimization-suites-bh--gb1-c5) | **C5** — downstream utility | NLS³-constructed substrates vs fixed-space specialists on BH & GB1 optimization |
| [`audit/`](#3-hybrid-audit-engine-c1c2) | **C1/C2** — constructed space + trust | The Hybrid-Audit engine: schema, proposer, auditor, open-form bench |

---

## Setup

```bash
pip install -r requirements.txt
```

LLM-backed arms read two environment variables (any OpenAI-compatible endpoint):

```bash
export NLSS_LLM_BASE_URL="https://your-endpoint/v1"
export NLSS_LLM_MODEL="your-model"
```

Baselines that never call an LLM (random, hill-climb, GP/BO family) run with
**no configuration at all** — everything in the quick-start below works
out of the box.

## Verify your install (≈2 min, no LLM)

```bash
bash smoke_test.sh
```

Runs the GB1 and BH random baselines plus import checks for all three
components. Expected final line: `SMOKE_OK`.

---

## 1. External relevance (HypoGeniC, C4)

Reproduces the paper's transfer evaluation: generate a hypothesis bank on a
dataset's train split with the official HypoGeniC loop, then measure held-out
accuracy on the test split.

```bash
python3 hypo/runners/run_hypogenic.py headline_binary
```

Datasets (MIT-licensed, from HypoBench; see `THIRD_PARTY.md`):
`deceptive_reviews` · `dreaddit` · `headline_binary` · `persuasive_pairs`

Reference numbers from the paper (held-out accuracy):

| deceptive_reviews | headline_binary | persuasive_pairs | dreaddit |
|:---:|:---:|:---:|:---:|
| 0.38 | 0.54 | 0.80 | 0.70 |

## 2. Downstream optimization suites (BH / GB1, C5)

The Buchwald–Hartwig (BH) reaction-yield and GB1 protein-fitness suites
compare NLS³-constructed substrates against fixed-space specialists
(GP-BO, BALLET, BO-DKL, Gryffin, CLADE, …) under a preregistered protocol.

```bash
# quick check (no LLM):
python3 scripts/run_bh_campaign.py  --arms random --seeds 1 --out work/bh_smoke
python3 scripts/run_gb1_campaign.py --arms random --seeds 1 --out work/gb1_smoke

# more arms:
#   BH:  nlss_graph, bo_gp, ballet_level, bo_dkl, gryffin_plain, random, hillclimb
#   GB1: nlss_hamming, gb1_gpbo, gb1_ballet, gb1_alde, random, hillclimb
python3 scripts/run_gb1_campaign.py --arms gb1_gpbo,random --seeds 5 --out work/gb1_5seed
```

The full preregistered protocol (frozen registry, Holm correction, TOST gates,
all arms × seeds) is `scripts/v8_run_bh_gb1.py` — see `--help`.

Full-pool GB1 numbers need the 149k-row measured fitness table
(not shipped; built-in demo subset keeps every arm runnable without it):

```bash
bash scripts/download_gb1_data.sh   # installs data/gb1/GB1.xlsx
```

BH ships with its data: `data/bh/data_table.csv` (MIT, doylelab).

## 3. Hybrid-Audit engine (C1/C2)

`audit/nlss_v10/` is the engine the paper's Construct → Explore → Evolve
pipeline runs on:

- `schema/` — the CHI substrate schema (`v10.1`) + operator executor
- `construct/proposer.py` — hypothesis proposal with content-addressed
  prompt provenance (`prompt_hash` covers the full rendered prompt)
- `audit/engine.py` — the Hybrid Auditor (7 score dimensions, hard
  obligations, LLM-judge client)
- `bench/openform_bench.py` — open-form benchmark driver
  (arms: `full` / `fixedmenu_or` / `fixedmenu_llm` / `open_noaudit`)

Minimal example (no LLM needed for schema checks):

```python
import sys; sys.path.insert(0, "audit")
from nlss_v10.schema.chi_schema import validate
from nlss_v10.audit.engine import HybridAuditor, SCORE_DIMS
# validate(chi_dict) -> ValidationResult
# HybridAuditor(judge).audit(chi)  — wire your LLM into audit/judge_client.py
```

---

## Repository layout

```
NLS3/
├── hypo/                  C4: vendored HypoGeniC (MIT) + C4 datasets + runner
├── nlss/                  C5: shared package (campaigns, stats, prereg, core)
├── scripts/               C5: campaign runners + full preregistered protocol
├── audit/nlss_v10/        C1/C2: Hybrid-Audit engine (schema/proposer/auditor/bench)
├── data/bh/               BH reaction data (MIT, shipped)
├── smoke_test.sh          2-minute no-LLM verification
├── THIRD_PARTY.md         per-component licenses and data provenance
└── requirements.txt
```

## What is intentionally not here

- Experiment outputs (runs, results, cost ledgers, preregistration snapshots)
- Internal endpoints, credentials, or model names — set them via env vars
- Upstream docs/assets not needed to run the code

## License

Our code: **MIT** (see `LICENSE`). Vendored third-party code and data keep
their upstream licenses — see [`THIRD_PARTY.md`](THIRD_PARTY.md) for the
per-component list (HypoGeniC: MIT; HypoBench: MIT; BH data: MIT; GB1:
download-on-demand).
