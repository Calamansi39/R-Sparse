import argparse
import json
import os
import sys
from pathlib import Path

from utils.setup import setup_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=None, help="Comma-separated task names.")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", default="1")
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--method", type=str, default="full")
    parser.add_argument("--target_sparsity", type=float, default=0.5)
    parser.add_argument("--prefill_ratio", type=float, default=0.1)
    parser.add_argument("--disable_prefill_protection", action="store_true")
    parser.add_argument("--sparse_ratio", type=float, default=1)
    parser.add_argument("--config_file", type=str, default="config/llama-2-7b-hf_default.json")
    parser.add_argument("--sparse_config_file", type=str, default=None)
    parser.add_argument("--arc_saved_dir", type=str, default=None)
    parser.add_argument("--arc_dataset", type=str, default="wikitext2")
    parser.add_argument("--arc_metric", type=str, default="max")
    parser.add_argument("--arc_quant_type", type=str, default="NVFP4")

    parser.add_argument(
        "--official_lm_eval_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "lm-evaluation-harness"),
        help="Path to the official lm-evaluation-harness checkout.",
    )
    return parser.parse_args()


def load_official_lm_eval(official_lm_eval_path: str):
    official_path = Path(official_lm_eval_path).resolve()
    if not official_path.exists():
        raise FileNotFoundError(f"official lm_eval path not found: {official_path}")
    sys.path.insert(0, str(official_path))
    from lm_eval import simple_evaluate
    from lm_eval.tasks import initialize_tasks
    from lm_eval.models.huggingface import HFLM

    return simple_evaluate, initialize_tasks, HFLM


def main():
    args = parse_args()
    if args.tasks is None:
        raise ValueError("--tasks is required for evaluation_official.py")

    simple_evaluate, initialize_tasks, HFLM = load_official_lm_eval(args.official_lm_eval_path)
    initialize_tasks()

    _, tokenizer, model = setup_model(args)
    model = model.eval().to(args.device)

    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        trust_remote_code=True,
        use_fast_tokenizer=False,
    )

    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = simple_evaluate(
        model=hflm,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        limit=args.limit,
        log_samples=False,
    )

    print(json.dumps(results["results"], indent=2, sort_keys=True))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
