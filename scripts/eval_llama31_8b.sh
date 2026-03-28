#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
MODEL="${MODEL:-/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B}"
LOW_RANK_DIR="${LOW_RANK_DIR:-../low_rank_models/llama-3.1-8b}"
CONFIG_FILE="${CONFIG_FILE:-config/llama-3.1-8b_default.json}"
SPARSE_CONFIG_FILE="${SPARSE_CONFIG_FILE:-config/llama3_sparsity_50_evolutionary_search.npy}"
PYTHON_BIN="${PYTHON_BIN:-/gemini/code/envs/r_sparse/bin/python}"

PPL_TASK="${PPL_TASK:-wikitext}"
MC_TASKS="${MC_TASKS:-arc_easy,arc_challenge,openbookqa,winogrande}"
MMLU_TASKS="${MMLU_TASKS:-hendrycksTest-*}"

PPL_FEWSHOT="${PPL_FEWSHOT:-0}"
MC_FEWSHOT="${MC_FEWSHOT:-0}"
MMLU_FEWSHOT="${MMLU_FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PREPARE_LOW_RANK="${PREPARE_LOW_RANK:-1}"

EXTRA_ARGS=()
if [[ -n "${CACHE_DIR:-}" ]]; then
  EXTRA_ARGS+=(--cache_dir "${CACHE_DIR}")
fi

# Resolve a Hugging Face cache root like
# /path/to/models--org--name into its active snapshot directory.
if [[ -f "${MODEL}/refs/main" && -d "${MODEL}/snapshots" ]]; then
  SNAPSHOT="$(<"${MODEL}/refs/main")"
  MODEL="${MODEL}/snapshots/${SNAPSHOT}"
fi

if [[ "${PREPARE_LOW_RANK}" == "1" ]]; then
  echo "Preparing low-rank weights for ${MODEL}"
  "${PYTHON_BIN}" -u utils/prepare_low_rank_weight.py \
    --model_name "${MODEL}" \
    --output_dir "${LOW_RANK_DIR}" \
    --skip_existing
else
  echo "Skipping low-rank preparation; using ${LOW_RANK_DIR}"
fi

echo "Running dense baseline on ${PPL_TASK}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluation.py \
  --tasks "${PPL_TASK}" \
  --num_fewshot "${PPL_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --model_name "${MODEL}" \
  --method full \
  "${EXTRA_ARGS[@]}"

echo "Running R-Sparse on ${PPL_TASK}"
echo "Using ${SPARSE_CONFIG_FILE}; this search file ships with Llama-3-8B and is only an approximation for Llama-3.1-8B."
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluation.py \
  --tasks "${PPL_TASK}" \
  --num_fewshot "${PPL_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --model_name "${MODEL}" \
  --method r_sparse \
  --sparse_config_file "${SPARSE_CONFIG_FILE}" \
  --config_file "${CONFIG_FILE}" \
  "${EXTRA_ARGS[@]}"

echo "Running dense baseline on ${MC_TASKS}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluation.py \
  --tasks "${MC_TASKS}" \
  --num_fewshot "${MC_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --model_name "${MODEL}" \
  --method full \
  "${EXTRA_ARGS[@]}"

echo "Running R-Sparse on ${MC_TASKS}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluation.py \
  --tasks "${MC_TASKS}" \
  --num_fewshot "${MC_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --model_name "${MODEL}" \
  --method r_sparse \
  --sparse_config_file "${SPARSE_CONFIG_FILE}" \
  --config_file "${CONFIG_FILE}" \
  "${EXTRA_ARGS[@]}"

echo "Running dense baseline on ${MMLU_TASKS}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluation.py \
  --tasks "${MMLU_TASKS}" \
  --num_fewshot "${MMLU_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --model_name "${MODEL}" \
  --method full \
  "${EXTRA_ARGS[@]}"

echo "Running R-Sparse on ${MMLU_TASKS}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluation.py \
  --tasks "${MMLU_TASKS}" \
  --num_fewshot "${MMLU_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --model_name "${MODEL}" \
  --method r_sparse \
  --sparse_config_file "${SPARSE_CONFIG_FILE}" \
  --config_file "${CONFIG_FILE}" \
  "${EXTRA_ARGS[@]}"
