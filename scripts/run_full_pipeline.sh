#!/usr/bin/env bash
# Full pipeline: HITL inference → LoRA training → Snowball evaluation
# Usage:
#   bash scripts/run_full_pipeline.sh [--clean]
#   tmux new -s pipeline 'bash scripts/run_full_pipeline.sh --clean 2>&1 | tee experiments/logs/pipeline_$(date +%Y%m%d_%H%M%S).log'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration (override via env vars) ──────────────
DATA_ROOT="${DATA_ROOT:-/home/kokyungmin/data/WB_WoB-ReID}"
SPLIT="${SPLIT:-both_large}"
N_QUERIES="${N_QUERIES:-1000}"
N_NEGATIVES="${N_NEGATIVES:-30}"
MIN_SAMPLES="${MIN_SAMPLES:-20}"
EPOCHS="${EPOCHS:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
REID_MODEL="${REID_MODEL:-arcface-dinov2}"
HITL_DIR="${HITL_DIR:-data/hitl}"
LORA_OUTPUT="${LORA_OUTPUT:-models/vlm_verifier_lora}"
EVAL_PAIRS="${EVAL_PAIRS:-data/eval_pairs_${SPLIT}.jsonl}"

CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
    esac
done

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# ── Pre-flight ─────────────────────────────────────────
# Create log directory BEFORE anything else (needed for tee in tmux)
mkdir -p experiments/logs

log "Pipeline starting from: $PROJECT_ROOT"
log "Data root   : $DATA_ROOT"
log "Split       : $SPLIT"
log "N queries   : $N_QUERIES"
log "N negatives : $N_NEGATIVES"
log "Min samples : $MIN_SAMPLES"
log "Epochs      : $EPOCHS"

if [[ "$CLEAN" == "1" ]]; then
    log "Cleaning previous HITL data in $HITL_DIR ..."
    rm -rf "$HITL_DIR"
fi

# ── Step 1: HITL VLM Inference ─────────────────────────
log "======== Step 1/3: HITL VLM Inference ========"
uv run python scripts/run_hitl_inference.py \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --hitl-dir "$HITL_DIR" \
    --n-queries "$N_QUERIES" \
    --n-negatives "$N_NEGATIVES"

# Verify Step 1 produced labeled data
if [[ ! -f "$HITL_DIR/labeled.jsonl" ]]; then
    log "ERROR: Step 1 produced no labeled data ($HITL_DIR/labeled.jsonl not found)."
    log "  Possible cause: no wrong VLM predictions, or person ID mismatch."
    exit 1
fi
N_LABELED=$(wc -l < "$HITL_DIR/labeled.jsonl")
log "Step 1 complete. Labeled samples: $N_LABELED"

# ── Step 2: LoRA Fine-tuning ──────────────────────────
log "======== Step 2/3: LoRA Fine-tuning ========"
uv run python scripts/lora_train.py \
    --labeled-jsonl "$HITL_DIR/labeled.jsonl" \
    --output-base "$LORA_OUTPUT" \
    --min-samples "$MIN_SAMPLES" \
    --epochs "$EPOCHS"

# Verify Step 2 produced adapter
if [[ ! -e "$LORA_OUTPUT/latest" ]]; then
    log "ERROR: Step 2 did not produce LoRA adapter ($LORA_OUTPUT/latest not found)."
    exit 1
fi
log "Step 2 complete. Adapter: $(readlink "$LORA_OUTPUT/latest")"

# ── Step 3: Snowball Evaluation ────────────────────────
log "======== Step 3/3: Snowball Evaluation ========"
uv run python scripts/evaluate_snowball.py \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --eval-pairs "$EVAL_PAIRS" \
    --reid-model "$REID_MODEL" \
    --n-queries "$N_QUERIES" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --lora-adapter "$LORA_OUTPUT/latest"
log "Step 3 complete."

log "======== Pipeline finished successfully ========"
