import argparse
import os
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
]


def robust_quantile(tensor, q, max_points=200000):
    flat = tensor.detach().reshape(-1)
    if flat.numel() > max_points:
        step = max(flat.numel() // max_points, 1)
        flat = flat[::step][:max_points]
    return float(torch.quantile(flat, q))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_root",
        default="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B",
    )
    parser.add_argument(
        "--low_rank_dir",
        default="/gemini/code/NMSparsity/low_rank_models/llama-3.1-8b",
    )
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--layers", default="0,15,31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="/gemini/code/NMSparsity/R-Sparse/results/rsparse_heatmap_llama31_wikitext.png",
    )
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--vmax_quantile",
        type=float,
        default=0.995,
        help="If --vmax is unset, use this per-subplot quantile as linear color upper bound.",
    )
    return parser.parse_args()


def resolve_model_path(model_root):
    ref_file = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")
    if os.path.isfile(ref_file) and os.path.isdir(snapshots_dir):
        with open(ref_file) as f:
            snapshot = f.read().strip()
        snapshot_dir = os.path.join(snapshots_dir, snapshot)
        if os.path.isdir(snapshot_dir):
            return snapshot_dir
    return model_root


def build_calibration_samples(tokenizer, args):
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    full_text = " ".join(ds[args.text_field])
    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids

    random.seed(args.seed)
    samples = []
    for _ in range(args.nsamples):
        start = random.randint(0, input_ids.shape[1] - args.seqlen - 1)
        end = start + args.seqlen
        samples.append(input_ids[:, start:end])
    return samples


def register_hooks(model, target_names, stats):
    handles = []

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0].detach().float().abs()
            stats[name]["sum"] += x.sum(dim=(0, 1)).cpu()
            stats[name]["count"] += x.shape[0] * x.shape[1]

        return hook

    for name, module in model.named_modules():
        if name in target_names:
            in_features = module.in_features
            stats[name] = {"sum": torch.zeros(in_features), "count": 0}
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    return handles


def build_target_names(layers):
    names = []
    for layer in layers:
        for module_name in TARGET_MODULES:
            names.append(f"model.layers.{layer}.{module_name}")
    return names


def contribution_matrix(mean_abs_input, svd_path):
    _, s, v, _ = torch.load(svd_path, map_location="cpu")
    s = s.float()
    v = v.float()
    contrib = (v.abs() * s.unsqueeze(0)).T
    contrib = contrib * mean_abs_input.unsqueeze(0)
    row_order = contrib.sum(dim=1).argsort()
    col_order = contrib.sum(dim=0).argsort()
    contrib = contrib[row_order][:, col_order]
    return contrib


def plot_heatmaps(mats, layers, output_path, vmin, vmax, vmax_quantile):
    fig, axes = plt.subplots(len(layers), len(TARGET_MODULES), figsize=(28, 11))
    if len(layers) == 1:
        axes = axes[None, :]

    for row, layer in enumerate(layers):
        for col, module_name in enumerate(TARGET_MODULES):
            ax = axes[row][col]
            key = f"model.layers.{layer}.{module_name}"
            mat = mats[key]
            local_vmax = vmax
            if local_vmax is None:
                local_vmax = robust_quantile(mat, vmax_quantile)
            norm = Normalize(vmin=vmin, vmax=local_vmax)
            im = ax.imshow(
                mat.detach().numpy(),
                cmap="coolwarm",
                aspect="auto",
                interpolation="nearest",
                norm=norm,
            )
            short_name = module_name.replace(".", "_")
            ax.set_title(f"Layer {layer}, {short_name}", fontsize=11)
            ax.set_xlabel("Input Channel Index")
            ax.set_ylabel("SVD Index")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    model_path = resolve_model_path(args.model_root)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    target_names = build_target_names(layers)
    stats = OrderedDict()
    handles = register_hooks(model, target_names, stats)

    samples = build_calibration_samples(tokenizer, args)
    with torch.no_grad():
        for sample in samples:
            model(sample.to(model.device))

    for handle in handles:
        handle.remove()

    mats = OrderedDict()
    for name in target_names:
        mean_abs_input = stats[name]["sum"] / max(stats[name]["count"], 1)
        svd_path = os.path.join(args.low_rank_dir, f"{name}.pt")
        mats[name] = contribution_matrix(mean_abs_input, svd_path)

    plot_heatmaps(mats, layers, args.output, args.vmin, args.vmax, args.vmax_quantile)
    print(f"saved heatmap: {args.output}")


if __name__ == "__main__":
    main()
