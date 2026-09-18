# Third-party components and data

| Component | Path here | Upstream | License | Shipped as |
|---|---|---|---|---|
| HypoGeniC | `hypo/hypogenic/` | github.com/ChicagoHAI/HypoGeniC | MIT (c) 2024 Haokun Liu, Chenfei Yuan, Yangqiaoyu Zhou, Tejes Srivastava and contributors — see `hypo/LICENSE.hypogenic` | Vendored package core (only `hypogenic/`; upstream docs/literature assets not included) |
| HypoBench datasets | `hypo/data/{deceptive_reviews,dreaddit,headline_binary,persuasive_pairs}` | github.com/chicago-ai-lab/HypoBench | MIT (c) 2025 Chicago Human+AI Lab — see `hypo/data/LICENSE.hypobench` | Only the four datasets used in the paper's external-relevance evaluation |
| Buchwald–Hartwig reaction data | `data/bh/data_table.csv` | doylelab.org / github.com/doylelab/rxnpredict | MIT (c) 2017 Ahneman, Estrada, Lin, Dreher, Doyle — see `data/bh/LICENSE.doylelab-rxnpredict` | Vendored (small, MIT) |
| GB1 measured fitness table | `data/gb1/GB1.xlsx` (NOT shipped) | CLADE (gb1_clade) repo | See upstream repository | Download via `scripts/download_gb1_data.sh`; a built-in demo subset keeps all code runnable without it |

LLM endpoints and model names are intentionally NOT configured in this repo.
Set `NLSS_LLM_BASE_URL` (OpenAI-compatible) and `NLSS_LLM_MODEL` to your own
endpoint; baselines that do not call an LLM (random, hill-climb, GP/BO family)
run with no configuration at all.
