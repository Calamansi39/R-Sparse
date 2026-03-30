# Llama-3.1-8B on arc_easy,boolq,piqa,rte (GPU 1)

- Model: `/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B`
- Dtype: `bf16`
- Sparse config: `/gemini/code/NMSparsity/R-Sparse/config/llama3_sparsity_50_evolutionary_search.npy`
- Config file: `/gemini/code/NMSparsity/R-Sparse/config/llama-3.1-8b_default.json`
- Batch size: `1`
- Few-shot: `0`

| Task | Metric | Dense | R-Sparse |
|---|---:|---:|---:|
| arc_easy | acc | 0.8148 | 0.7917 |
| arc_easy | acc_norm | 0.8110 | 0.7854 |
| boolq | acc | 0.8211 | 0.7801 |
| piqa | acc | 0.8014 | 0.7954 |
| piqa | acc_norm | 0.8123 | 0.8128 |
| rte | acc | 0.6895 | 0.6715 |
