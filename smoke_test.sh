#!/bin/bash
# Runs the three no-LLM smoke checks (~2 min total):
#   1. GB1 random baseline (uses built-in demo subset if full xlsx absent)
#   2. BH random baseline (data ships in data/bh/)
#   3. import checks for the Hybrid-Audit engine and the vendored HypoGeniC
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export NLSS_BH_DATA_TABLE="$HERE/data/bh/data_table.csv"
export NLSS_LLM_BASE_URL="${NLSS_LLM_BASE_URL:-}"
export NLSS_LLM_MODEL="${NLSS_LLM_MODEL:-}"

echo "[1/3] GB1 random baseline..."
python3 scripts/run_gb1_campaign.py --arms random --seeds 1 --n-initial 8 --batch 4 --rounds 2 --out work/smoke_gb1 | tail -2

echo "[2/3] BH random baseline..."
python3 scripts/run_bh_campaign.py --arms random --seeds 1 --n-initial 8 --batch 4 --rounds 3 --out work/smoke_bh | tail -2

echo "[3/3] import checks..."
python3 - << 'PY'
import sys
sys.path.insert(0, "audit"); sys.path.insert(0, "hypo")
from nlss_v10.schema.chi_schema import validate, SCHEMA_VERSION
from nlss_v10.construct.proposer import Proposer, render_task
from nlss_v10.audit.engine import HybridAuditor, SCORE_DIMS
from hypogenic.algorithm.generation import DefaultGeneration
from hypogenic.algorithm.inference import DefaultInference
from hypogenic.extract_label import extract_label_register
from hypogenic.LLM_wrapper import llm_wrapper_register
print(f"  nlss_v10 schema {SCHEMA_VERSION}, {len(SCORE_DIMS)} audit dims; hypogenic OK")
PY
echo "SMOKE_OK — all three components verified."
