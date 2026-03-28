import argparse
import csv
import json
import os
import random
import sys
from types import SimpleNamespace

import torch
from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.modeling_llama import R_Sparse_Linear
from utils.setup import setup_model


TARGET_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_root",
        default="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B",
    )
    parser.add_argument(
        "--config_file",
        default="config/llama-3.1-8b_default.json",
    )
    parser.add_argument(
        "--sparse_config_file",
        default="config/llama3_sparsity_50_evolutionary_search.npy",
    )
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--nsamples", type=int, default=5)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="0,15,31")
    parser.add_argument("--prefill_ratio", type=float, default=0.1)
    parser.add_argument(
        "--output_json",
        default="/gemini/code/NMSparsity/R-Sparse/results/rsparse_sparsity_stats_layers_0_15_31.json",
    )
    parser.add_argument(
        "--output_csv",
        default="/gemini/code/NMSparsity/R-Sparse/results/rsparse_sparsity_stats_layers_0_15_31.csv",
    )
    return parser.parse_args()


def build_samples(tokenizer, args):
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    full_text = " ".join(ds[args.text_field])
    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids

    random.seed(args.seed)
    samples = []
    for sample_id in range(args.nsamples):
        start = random.randint(0, input_ids.shape[1] - args.seqlen - 1)
        end = start + args.seqlen
        samples.append(
            {
                "sample_id": sample_id,
                "start": start,
                "end": end,
                "input_ids": input_ids[:, start:end],
            }
        )
    return samples


class SparsityRecorder:
    def __init__(self, layers):
        self.layers = layers
        self.current_sample_id = None
        self.stats = {}

    def set_sample(self, sample_id):
        self.current_sample_id = sample_id
        self.stats.setdefault(sample_id, {})

    def hook(self, name):
        def fn(module, inputs):
            if self.current_sample_id is None:
                return
            if module.flag_getting_threshold:
                return
            if module.mode not in ("sparse", "r_sparse"):
                return

            x = inputs[0].detach()
            num_tokens = x.size(1)

            if module.prefill_ratio == 1:
                if num_tokens > 1:
                    return
                sparse_x = x
            else:
                if num_tokens <= 1:
                    return
                dense_tokens = int(num_tokens * (1 - module.prefill_ratio))
                sparse_x = x[:, dense_tokens:, :]
                if sparse_x.numel() == 0:
                    return

            if module.mode == "sparse":
                s_mask = sparse_x.abs().gt(module.threshold)
            else:
                scale_input = sparse_x * module.scale.unsqueeze(0).unsqueeze(0)
                s_mask = scale_input.abs().gt(module.threshold)

            zero_count = int((~s_mask).sum().item())
            total_count = int(s_mask.numel())
            kept_count = total_count - zero_count

            layer_id = int(name.split(".")[2])
            layer_bucket = self.stats[self.current_sample_id].setdefault(
                layer_id,
                {
                    "zero_count": 0,
                    "kept_count": 0,
                    "total_count": 0,
                    "modules": {},
                },
            )
            layer_bucket["zero_count"] += zero_count
            layer_bucket["kept_count"] += kept_count
            layer_bucket["total_count"] += total_count
            layer_bucket["modules"][name] = {
                "zero_count": zero_count,
                "kept_count": kept_count,
                "total_count": total_count,
                "sparsity": zero_count / total_count,
                "threshold": float(module.threshold),
                "target_sparsity": float(module.target_sparsity),
                "mode": module.mode,
            }

        return fn


def summarize(recorder, samples, layers):
    summary = {
        "samples": [],
        "aggregate": {
            "layers": {},
            "overall": {},
        },
    }

    layer_values = {layer: [] for layer in layers}
    overall_values = []

    for sample in samples:
        sample_id = sample["sample_id"]
        sample_stats = recorder.stats.get(sample_id, {})
        sample_entry = {
            "sample_id": sample_id,
            "start": sample["start"],
            "end": sample["end"],
            "layers": {},
        }

        overall_zero = 0
        overall_total = 0

        for layer in layers:
            layer_stat = sample_stats.get(
                layer,
                {"zero_count": 0, "kept_count": 0, "total_count": 0, "modules": {}},
            )
            total = layer_stat["total_count"]
            sparsity = layer_stat["zero_count"] / total if total else 0.0
            sample_entry["layers"][str(layer)] = {
                "sparsity": sparsity,
                "zero_count": layer_stat["zero_count"],
                "kept_count": layer_stat["kept_count"],
                "total_count": total,
                "modules": layer_stat["modules"],
            }
            layer_values[layer].append(sparsity)
            overall_zero += layer_stat["zero_count"]
            overall_total += total

        overall_sparsity = overall_zero / overall_total if overall_total else 0.0
        sample_entry["overall"] = {
            "sparsity": overall_sparsity,
            "zero_count": overall_zero,
            "total_count": overall_total,
        }
        overall_values.append(overall_sparsity)
        summary["samples"].append(sample_entry)

    for layer in layers:
        values = layer_values[layer]
        mean = sum(values) / len(values) if values else 0.0
        std = (
            (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            if values
            else 0.0
        )
        summary["aggregate"]["layers"][str(layer)] = {
            "mean_sparsity": mean,
            "std_sparsity": std,
            "min_sparsity": min(values) if values else 0.0,
            "max_sparsity": max(values) if values else 0.0,
        }

    overall_mean = sum(overall_values) / len(overall_values) if overall_values else 0.0
    overall_std = (
        (sum((v - overall_mean) ** 2 for v in overall_values) / len(overall_values)) ** 0.5
        if overall_values
        else 0.0
    )
    summary["aggregate"]["overall"] = {
        "mean_sparsity": overall_mean,
        "std_sparsity": overall_std,
        "min_sparsity": min(overall_values) if overall_values else 0.0,
        "max_sparsity": max(overall_values) if overall_values else 0.0,
    }
    return summary


def write_csv(summary, output_csv, layers):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "scope", "layer", "sparsity", "zero_count", "total_count"])
        for sample in summary["samples"]:
            for layer in layers:
                layer_stat = sample["layers"][str(layer)]
                writer.writerow(
                    [
                        sample["sample_id"],
                        "layer",
                        layer,
                        layer_stat["sparsity"],
                        layer_stat["zero_count"],
                        layer_stat["total_count"],
                    ]
                )
            writer.writerow(
                [
                    sample["sample_id"],
                    "overall",
                    "all",
                    sample["overall"]["sparsity"],
                    sample["overall"]["zero_count"],
                    sample["overall"]["total_count"],
                ]
            )


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    target_names = [
        f"model.layers.{layer}.{module_name}"
        for layer in layers
        for module_name in TARGET_MODULES
    ]

    model_args = SimpleNamespace(
        model_name=args.model_root,
        cache_dir=None,
        device="cuda:0",
        method="r_sparse",
        target_sparsity=0.5,
        prefill_ratio=args.prefill_ratio,
        sparse_ratio=1,
        config_file=args.config_file,
        sparse_config_file=args.sparse_config_file,
    )

    _, tokenizer, model = setup_model(model_args)
    model = model.eval()

    recorder = SparsityRecorder(layers)
    handles = []
    for name, module in model.named_modules():
        if name in target_names and isinstance(module, R_Sparse_Linear):
            handles.append(module.register_forward_pre_hook(recorder.hook(name)))

    samples = build_samples(tokenizer, args)
    with torch.no_grad():
        for sample in samples:
            recorder.set_sample(sample["sample_id"])
            model(sample["input_ids"].to(model.device))

    for handle in handles:
        handle.remove()

    summary = summarize(recorder, samples, layers)
    summary["config"] = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "layers": layers,
        "prefill_ratio": args.prefill_ratio,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    write_csv(summary, args.output_csv, layers)
    print(f"saved json: {args.output_json}")
    print(f"saved csv: {args.output_csv}")
    for layer in layers:
        agg = summary["aggregate"]["layers"][str(layer)]
        print(
            f"layer {layer}: mean={agg['mean_sparsity']:.6f}, "
            f"std={agg['std_sparsity']:.6f}, "
            f"min={agg['min_sparsity']:.6f}, max={agg['max_sparsity']:.6f}"
        )
    overall = summary["aggregate"]["overall"]
    print(
        f"overall: mean={overall['mean_sparsity']:.6f}, "
        f"std={overall['std_sparsity']:.6f}, "
        f"min={overall['min_sparsity']:.6f}, max={overall['max_sparsity']:.6f}"
    )


if __name__ == "__main__":
    main()
