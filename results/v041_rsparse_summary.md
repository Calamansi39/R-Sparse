# R-Sparse v0.4.1 Summary

Model: `meta-llama/Llama-3.1-8B`
Method: `R-Sparse`
Sparsity config: `config/llama3_sparsity_50_evolutionary_search.npy`
Harness: `lm-evaluation-harness v0.4.1`

Notes:
- `PPL` below uses token-level `wikitext2` perplexity, matching the previously agreed 6-7 scale.
- The official `lm_eval v0.4.1` `wikitext` task reports `word_perplexity=9.943640368466278`; that is a different metric and is not used as the final `PPL` here.
- `MMLU` is reported with `lm_eval v0.4.1` task-default `0-shot` because the task config does not encode a built-in few-shot count.

| Benchmark | Metric | Value |
|---|---|---:|
| PPL | token-level ppl | 6.350313 |
| ARC_C | acc | 0.482082 |
| ARC_C | acc_norm | 0.511945 |
| MMLU | acc | 0.605469 |
| WG | acc | 0.707182 |
| OBQA | acc | 0.338000 |
| OBQA | acc_norm | 0.466000 |
| ARC_E | acc | 0.797138 |
| ARC_E | acc_norm | 0.786616 |
| BOOLQ | acc | 0.780734 |
| PIQA | acc | 0.792709 |
| PIQA | acc_norm | 0.803047 |
| RTE | acc | 0.682310 |

Artifacts:
- `results/v041_rsparse_wikitext.json`
- `results/v041_rsparse_mc.json`
- `results/v041_rsparse_mmlu_gpu3.json`
