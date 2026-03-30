# RTE comparison: official lm-eval vs R-Sparse repo

## Official lm-eval
- command family: `lm_eval run --model hf`
- result: `0.7112`
- output: `/gemini/code/NMSparsity/R-Sparse/results/lm_eval_official_rte_llama31_gpu1_2026-03-12T14-40-04.262567.json`
- task config source: `/gemini/code/tmp/lm-evaluation-harness/lm_eval/tasks/glue/rte/default.yaml`
- dataset_path: `nyu-mll/glue`
- validation docs: `277`
- metric: `acc`
- num_fewshot: `0`
- prompt: `{{sentence1}}\nQuestion: {{sentence2}} True or False?\nAnswer:`
- choices: `["True", "False"]`
- dtype: `bfloat16`

## Official lm-eval with slow tokenizer
- result: `0.7112`
- model_args extra: `use_fast_tokenizer=False`

## R-Sparse repo dense baseline
- result: `0.6895`
- output: `/gemini/code/NMSparsity/R-Sparse/results/llama31_rsparse_arc_easy_boolq_piqa_rte_gpu1.log`
- task implementation: `/gemini/code/NMSparsity/R-Sparse/lm_eval/tasks/glue.py`
- dataset_path: `glue`
- validation docs: `277`
- metric: `acc`
- num_fewshot: `0`
- prompt: `{sentence1}\nQuestion: {sentence2} True or False?\nAnswer:`
- tokenizer load in setup: `use_fast=False`
- dtype: inherited from HF load path in repo setup

## High-confidence findings
- `rte` here is `acc`, not `acc_norm`.
- The prompt text matches between official and R-Sparse implementations.
- `num_fewshot=0` matches.
- Switching official lm-eval to slow tokenizer does **not** change the `0.7112` result.
- The remaining gap is therefore not explained by `acc` vs `acc_norm`, prompt wording, or fast/slow tokenizer.
- The most likely source is the old forked `lm_eval`/HF backend and legacy task stack used inside R-Sparse.
