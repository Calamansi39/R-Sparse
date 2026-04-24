#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:?output dir required}"
MODEL_PATH="${2:-/home/wmq/models/llama-3.1-8b}"
CONFIG_FILE="${3:-/mnt/data2/lbc/R-Sparse/config/llama-3.1-8b_default.json}"
SOURCE_SEARCH_FILE="${4:-/mnt/data2/lbc/R-Sparse/config/llama3_sparsity_50_evolutionary_search.npy}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/anaconda3/bin/conda run -n smoothquant python}"

mkdir -p "$OUT_DIR"

S="0.50"
TAG="50"
LOG_PATH="$OUT_DIR/search_${TAG}.log"

echo "[$(date --iso-8601=seconds)] target_sparsity=$S" | tee -a "$LOG_PATH"
eval "$PYTHON_BIN" /mnt/data2/lbc/R-Sparse/scripts/search_rsparse_alpha_grid.py \
  --model_path "$MODEL_PATH" \
  --config_file "$CONFIG_FILE" \
  --target_sparsity "$S" \
  --source_search_file "$SOURCE_SEARCH_FILE" \
  --output_search_file "$OUT_DIR/llama31_sparsity_${TAG}_grid_search.npy" \
  --output_metadata_json "$OUT_DIR/llama31_sparsity_${TAG}_grid_search.json" \
  >> "$LOG_PATH" 2>&1
