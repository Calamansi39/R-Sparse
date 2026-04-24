import argparse
import math
import os
import sys

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.setup import setup_model
from utils.linear_input_stats import (
    LinearInputStatsLogger,
    clear_linear_input_stats_logger,
    dump_linear_input_stats_logger,
    set_linear_input_stats_logger,
)


def load_wikitext_split(split: str):
    cache_dir = (
        "/home/wmq/.cache/huggingface/datasets/wikitext/"
        "wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"
    )
    arrow_path = os.path.join(cache_dir, f"wikitext-{split}.arrow")
    if os.path.isfile(arrow_path):
        return Dataset.from_file(arrow_path)
    return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)


def build_text(split: str, max_samples: int):
    dataset = load_wikitext_split(split)
    chunks = []
    used = 0
    for row in dataset:
        text = row["text"]
        if not text or not text.strip():
            continue
        chunks.append(text)
        used += 1
        if max_samples > 0 and used >= max_samples:
            break
    return "\n\n".join(chunks)


def eval_token_ppl(model, tokenizer, text: str, context_size: int, window_size: int, device: str):
    encodings = tokenizer(text, return_tensors="pt")
    stride = window_size
    max_length = context_size + window_size
    seq_len = encodings.input_ids.size(1)
    seq_len = seq_len - (seq_len % stride)
    if seq_len <= 0:
        raise ValueError("No tokens available for evaluation.")

    nlls = []
    model.eval()
    for begin_loc in tqdm(range(0, seq_len, stride), desc="PPL"):
        end_loc = begin_loc + max_length
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=target_ids)
            nlls.append(outputs.loss.float())
        if end_loc >= seq_len:
            break

    return torch.exp(torch.stack(nlls).double().mean()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--method", type=str, default="r_sparse")
    parser.add_argument("--target_sparsity", type=float, default=0.5)
    parser.add_argument("--prefill_ratio", type=float, default=0.1)
    parser.add_argument("--disable_prefill_protection", action="store_true")
    parser.add_argument("--sparse_ratio", type=float, default=1)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--sparse_config_file", type=str, default=None)
    parser.add_argument("--arc_saved_dir", type=str, default=None)
    parser.add_argument("--arc_dataset", type=str, default="wikitext2")
    parser.add_argument("--arc_metric", type=str, default="max")
    parser.add_argument("--arc_quant_type", type=str, default="NVFP4")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--context_size", type=int, default=2048)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stats_json", type=str, default=None)
    args = parser.parse_args()

    _, tokenizer, model = setup_model(args)
    model = model.eval().to(args.device)
    text = build_text(args.split, args.max_samples)
    if args.stats_json:
        set_linear_input_stats_logger(
            LinearInputStatsLogger(
                args.stats_json,
                seq_len=args.context_size + args.window_size,
            )
        )
    ppl = eval_token_ppl(
        model=model,
        tokenizer=tokenizer,
        text=text,
        context_size=args.context_size,
        window_size=args.window_size,
        device=args.device,
    )
    if args.stats_json:
        dump_linear_input_stats_logger()
        clear_linear_input_stats_logger()
    print(f"TOKEN_PPL: {ppl:.6f}")


if __name__ == "__main__":
    main()
