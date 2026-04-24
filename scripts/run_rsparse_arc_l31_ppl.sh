#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-5}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
CONFIG_FILE="${CONFIG_FILE:-/mnt/data2/lbc/R-Sparse/config/llama-3.1-8b_default.json}"
SPARSE_CONFIG_FILE="${SPARSE_CONFIG_FILE:-/mnt/data2/lbc/R-Sparse/config/llama3_sparsity_50_evolutionary_search.npy}"
ARC_SAVED_DIR="${ARC_SAVED_DIR:-}"
ARC_DATASET="${ARC_DATASET:-wikitext2}"
ARC_METRIC="${ARC_METRIC:-max}"
ARC_QUANT_TYPE="${ARC_QUANT_TYPE:-NVFP4}"
DTYPE="${DTYPE:-bfloat16}"
LOW_RANK_DIR="${LOW_RANK_DIR:-/mnt/data2/lbc/low_rank_models/llama-3.1-8b}"
PREPARE_LOW_RANK="${PREPARE_LOW_RANK:-1}"
DISABLE_PREFILL_PROTECTION="${DISABLE_PREFILL_PROTECTION:-0}"

cd /mnt/data2/lbc/R-Sparse

if [[ "${PREPARE_LOW_RANK}" == "1" ]]; then
  PYTHONNOUSERSITE=1 python -u utils/prepare_low_rank_weight.py \
    --model_name "${MODEL}" \
    --output_dir "${LOW_RANK_DIR}" \
    --torch_dtype "${DTYPE}" \
    --skip_existing
fi

ARGS=(
  --tasks wikitext
  --batch_size 1
  --model_name "${MODEL}"
  --method r_sparse
  --config_file "${CONFIG_FILE}"
  --sparse_config_file "${SPARSE_CONFIG_FILE}"
  --torch_dtype "${DTYPE}"
)

if [[ -n "${ARC_SAVED_DIR}" ]]; then
  ARGS+=(
    --arc_saved_dir "${ARC_SAVED_DIR}"
    --arc_dataset "${ARC_DATASET}"
    --arc_metric "${ARC_METRIC}"
    --arc_quant_type "${ARC_QUANT_TYPE}"
  )
fi

if [[ "${DISABLE_PREFILL_PROTECTION}" == "1" ]]; then
  ARGS+=(--disable_prefill_protection)
fi

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 python -u evaluation.py "${ARGS[@]}"
