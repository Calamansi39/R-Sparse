import os 
import torch
import argparse
import torch.nn as nn
import re
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer ids and ranges to generate, e.g. 22-31 or 0,3,8-10",
    )
    parser.add_argument(
        "--module_regex",
        type=str,
        default=None,
        help="Optional regex filter on linear module names.",
    )
    return parser.parse_args()


def parse_dtype(dtype_name):
    key = dtype_name.lower()
    if key in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if key in ["fp16", "float16", "half"]:
        return torch.float16
    if key in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def parse_layers(spec):
    if not spec:
        return None
    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def extract_layer_idx(module_name):
    match = re.search(r"\bmodel\.layers\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    return None

def main():
    args = parse_args()
    factor_dtype = parse_dtype(args.torch_dtype)
    selected_layers = parse_layers(args.layers)
    module_pattern = re.compile(args.module_regex) if args.module_regex else None
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=factor_dtype)
    os.makedirs(args.output_dir, exist_ok=True)
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            layer_idx = extract_layer_idx(name)
            if selected_layers is not None and layer_idx not in selected_layers:
                continue
            if module_pattern is not None and module_pattern.search(name) is None:
                continue
            output_path = os.path.join(args.output_dir, name + '.pt')
            if args.skip_existing and os.path.exists(output_path):
                print(f"Skipping existing low-rank factors: {name}")
                continue

            weight = m.weight.detach().to(torch.float32).cpu()
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            weight_reconstructed = (u * s.unsqueeze(0)) @ vh

            u = u.to(factor_dtype)
            s = s.to(factor_dtype)
            v = vh.T.to(factor_dtype)

            scale = v @ torch.diag(s)
            scale = scale.norm(dim=1)

            error = torch.norm(weight - weight_reconstructed)
            torch.save((u.cpu(), s.cpu(), v.cpu(), scale.cpu()), output_path)
            print(f"Saved low-rank factors: {name} | reconstruction_residual={error.item():.6e}")

if __name__ == '__main__':
    main()
