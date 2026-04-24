#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


RSPECS = [
    ("self_attn", "q_proj", "q", "q_svd_path"),
    ("self_attn", "k_proj", "k", "k_svd_path"),
    ("self_attn", "v_proj", "v", "v_svd_path"),
    ("self_attn", "o_proj", "o", "o_svd_path"),
    ("mlp", "gate_proj", "gate", "gate_svd_path"),
    ("mlp", "up_proj", "up", "up_svd_path"),
    ("mlp", "down_proj", "down", "down_svd_path"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--target_sparsity", type=float, required=True)
    parser.add_argument("--output_search_file", required=True)
    parser.add_argument("--output_metadata_json", required=True)
    parser.add_argument("--source_search_file", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens_per_module", type=int, default=16)
    parser.add_argument("--alpha_candidates", default="0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.0")
    return parser.parse_args()


def parse_dtype(name: str):
    key = name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def resolve_model_path(model_name: str):
    if os.path.isdir(model_name):
        ref_file = os.path.join(model_name, "refs", "main")
        snapshots_dir = os.path.join(model_name, "snapshots")
        if os.path.isfile(ref_file) and os.path.isdir(snapshots_dir):
            snapshot = Path(ref_file).read_text().strip()
            snapshot_dir = os.path.join(snapshots_dir, snapshot)
            if os.path.isdir(snapshot_dir):
                return snapshot_dir
    return model_name


def resolve_svd_path(config_file: str, path: str):
    if os.path.isabs(path):
        return path
    config_dir = os.path.dirname(os.path.abspath(config_file))
    resolved = os.path.abspath(os.path.join(config_dir, path))
    if os.path.exists(resolved):
        return resolved
    if "low_rank_models/" in path:
        tail = path.split("low_rank_models/", 1)[-1]
        fallback = os.path.abspath(os.path.join(config_dir, "..", "..", "low_rank_models", tail))
        if os.path.exists(fallback):
            return fallback
    return resolved


def projection_dims(hidden_size, intermediate_size, component, projection):
    if component == "self_attn":
        return hidden_size, hidden_size
    if projection == "down_proj":
        return intermediate_size, hidden_size
    return hidden_size, intermediate_size


def compute_budgeted_rank(in_features, out_features, target_sparsity, alpha):
    channels = max(int(in_features * (1 - target_sparsity) * alpha), 1)
    overall_budget = in_features * out_features * (1 - target_sparsity)
    sparse_budget = channels * out_features
    low_rank_budget = max(overall_budget - sparse_budget, 0.0)
    if low_rank_budget <= 0:
        return 0
    return max(int(low_rank_budget / (in_features + out_features)), 1)


class ReservoirSampler:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.count = 0
        self.buffer = None

    def update(self, x: torch.Tensor):
        x = x.detach().reshape(-1, x.shape[-1]).to("cpu", dtype=torch.float32)
        if x.numel() == 0:
            return
        if self.buffer is None:
            take = min(self.max_tokens, x.shape[0])
            self.buffer = x[:take].clone()
            self.count = x.shape[0]
            for idx in range(take, x.shape[0]):
                j = random.randint(0, self.count)
                if j < self.max_tokens:
                    self.buffer[j] = x[idx]
                self.count += 1
            return

        for idx in range(x.shape[0]):
            j = random.randint(0, self.count)
            if j < self.max_tokens:
                self.buffer[j] = x[idx]
            self.count += 1

    def tensor(self):
        return self.buffer.clone() if self.buffer is not None else None


def module_names(num_layers: int):
    for layer_idx in range(num_layers):
        for component, projection, short_name, path_key in RSPECS:
            yield layer_idx, component, projection, short_name, path_key, f"model.layers.{layer_idx}.{component}.{projection}"


def build_samples(tokenizer, args):
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    text = " ".join(ds[args.text_field])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids
    random.seed(args.seed)
    samples = []
    for _ in range(args.nsamples):
        start = random.randint(0, ids.shape[1] - args.seqlen - 1)
        samples.append(ids[:, start : start + args.seqlen])
    return samples


def choose_threshold(values: torch.Tensor, target_sparsity: float):
    flat = values.reshape(-1)
    kth = int(flat.numel() * target_sparsity)
    if kth <= 0:
        return None
    kth = min(kth, flat.numel() - 1)
    threshold = torch.topk(flat, kth, largest=False).values[-1]
    return threshold


def candidate_list(alpha_candidates: str, source_alpha: float | None):
    vals = [float(x) for x in alpha_candidates.split(",") if x.strip()]
    if source_alpha is not None:
        vals.append(float(source_alpha))
        for delta in (0.05, -0.05, 0.1, -0.1):
            vals.append(min(1.0, max(0.02, float(source_alpha) + delta)))
    vals = sorted({round(v, 6) for v in vals if 0.0 < v <= 1.0})
    return vals


def main():
    args = parse_args()
    torch_dtype = parse_dtype(args.dtype)
    model_path = resolve_model_path(args.model_path)

    with open(args.config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    source_search = None
    if args.source_search_file:
        source_search = torch.tensor(
            torch.from_numpy(__import__("numpy").loadtxt(args.source_search_file)).float().tolist()
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()

    samplers = {}
    handles = []
    for layer_idx, component, projection, short_name, path_key, name in module_names(model.config.num_hidden_layers):
        samplers[name] = ReservoirSampler(args.tokens_per_module)
        module = dict(model.named_modules())[name]
        handles.append(module.register_forward_pre_hook(lambda mod, inputs, n=name: samplers[n].update(inputs[0])))

    samples = build_samples(tokenizer, args)
    with torch.no_grad():
        for sample in samples:
            model(input_ids=sample.to(args.device))

    for handle in handles:
        handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import numpy as np

    search_values = []
    metadata = {
        "target_sparsity": args.target_sparsity,
        "source_search_file": args.source_search_file,
        "tokens_per_module": args.tokens_per_module,
        "nsamples": args.nsamples,
        "alpha_candidates": args.alpha_candidates,
        "modules": [],
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    ).eval()
    named_modules = dict(model.named_modules())

    source_alpha_values = None
    if args.source_search_file:
        source_alpha_values = np.loadtxt(args.source_search_file)[0::2]

    module_index = 0
    hidden = int(model.config.hidden_size)
    intermediate = int(model.config.intermediate_size)

    for layer_idx, component, projection, short_name, path_key, name in module_names(model.config.num_hidden_layers):
        x = samplers[name].tensor()
        if x is None:
            raise RuntimeError(f"No calibration activations captured for {name}")

        module = named_modules[name]
        weight = module.weight.detach().to(torch.float32)
        dense_x = x.to(torch.float32)
        dense_out = F.linear(dense_x, weight)

        in_features, out_features = projection_dims(hidden, intermediate, component, projection)
        source_alpha = None if source_alpha_values is None else float(source_alpha_values[module_index])
        candidates = candidate_list(args.alpha_candidates, source_alpha)

        svd_path = resolve_svd_path(args.config_file, config_data[path_key][layer_idx])
        u, s, v, scale = torch.load(svd_path, map_location="cpu")
        u = u.to(torch.float32)
        s = s.to(torch.float32)
        v = v.to(torch.float32)
        scale = scale.to(torch.float32)

        best = None
        for alpha in candidates:
            target_eff = 1 - (1 - args.target_sparsity) * alpha
            rank = compute_budgeted_rank(in_features, out_features, args.target_sparsity, alpha)

            if alpha >= 0.999:
                threshold_values = dense_x.abs()
            else:
                threshold_values = (dense_x * scale.unsqueeze(0)).abs()

            threshold = choose_threshold(threshold_values, target_eff)
            if threshold is None:
                zero_ratio = 0.0
                sparse_mask = torch.ones_like(dense_x)
            else:
                sparse_mask = threshold_values.gt(threshold).to(dense_x.dtype)
                zero_ratio = float((1.0 - sparse_mask).mean().item())

            sparse_input = dense_x * sparse_mask
            approx_out = F.linear(sparse_input, weight)

            if alpha < 0.999 and rank > 0:
                vs = v[:, :rank] * s[:rank].unsqueeze(0)
                low_rank_input = dense_x * (1.0 - sparse_mask)
                approx_out = approx_out + (low_rank_input @ vs @ u[:, :rank].transpose(0, 1))

            rel_mse = float(((dense_out - approx_out).pow(2).mean() / (dense_out.pow(2).mean() + 1e-8)).item())
            candidate = {
                "alpha": float(alpha),
                "rank": int(rank),
                "target_sparsity": float(target_eff),
                "estimated_zero_ratio": zero_ratio,
                "relative_mse": rel_mse,
            }
            if best is None or candidate["relative_mse"] < best["relative_mse"]:
                best = candidate

        search_values.extend([best["alpha"], float(args.target_sparsity)])
        metadata["modules"].append(
            {
                "module_index": module_index,
                "name": name,
                "source_alpha": source_alpha,
                **best,
            }
        )
        module_index += 1

    output_search = Path(args.output_search_file)
    output_search.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_search, np.asarray(search_values, dtype=np.float64), fmt="%.18e")

    output_meta = Path(args.output_metadata_json)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
