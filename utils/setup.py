import os
import sys
import json
import tqdm
import torch
import random
import datasets
import numpy as np 

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.modeling_llama import LlamaForCausalLM_R_Sparse, R_Sparse_Linear

__all__ = ['setup_config', 'setup_model']


def parse_dtype(dtype_name):
    if dtype_name is None:
        return None
    key = dtype_name.lower()
    if key in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if key in ["fp16", "float16", "half"]:
        return torch.float16
    if key in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_model_path(model_name):
    if os.path.isdir(model_name):
        ref_file = os.path.join(model_name, "refs", "main")
        snapshots_dir = os.path.join(model_name, "snapshots")
        if os.path.isfile(ref_file) and os.path.isdir(snapshots_dir):
            with open(ref_file) as f:
                snapshot = f.read().strip()
            snapshot_dir = os.path.join(snapshots_dir, snapshot)
            if os.path.isdir(snapshot_dir):
                return snapshot_dir
    return model_name


def _resolve_relative_to(base_file, path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    resolved = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(base_file)), path))
    if os.path.exists(resolved):
        return resolved
    if "low_rank_models/" in path:
        tail = path.split("low_rank_models/", 1)[-1]
        fallback_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "low_rank_models")
        )
        fallback = os.path.join(fallback_root, tail)
        if os.path.exists(fallback):
            return fallback
    return resolved

def setup_model(args):
    model_name = resolve_model_path(args.model_name)
    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            cache_dir=args.cache_dir,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            cache_dir=args.cache_dir,
        )
    torch_dtype = parse_dtype(getattr(args, "torch_dtype", None))
    common_kwargs = dict(
        config=config,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    if torch_dtype is not None:
        common_kwargs["torch_dtype"] = torch_dtype
    if args.method == 'full':
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    elif args.method == 'relufiction':
        config.hidden_act = 'relu'
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    elif args.method == 'r_sparse':
        config = setup_config(config, args)
        common_kwargs["config"] = config
        model = LlamaForCausalLM_R_Sparse.from_pretrained(model_name, **common_kwargs)
        model = set_threshold_r_sparse(model, config, args, R_Sparse_Linear, tokenizer)
    else:
        raise NotImplementedError
    return config, tokenizer, model



def set_threshold_r_sparse(model, config, args, module_type, tokenizer):
    if args.sparse_config_file is not None:
        sparse_config_file = np.loadtxt(args.sparse_config_file)
        index = 0
        for module_name, module in model.named_modules():
            if isinstance(module, module_type):
                if 'self_attn' in module_name:
                    in_channel = config.hidden_size
                    out_channel = config.hidden_size
                else:
                    if 'down_proj' in module_name:
                        in_channel = config.intermediate_size
                        out_channel = config.hidden_size
                    else:
                        in_channel = config.hidden_size
                        out_channel = config.intermediate_size

                alpha = sparse_config_file[index]
                s = sparse_config_file[index + 1]
                index += 2
                module.flag_getting_threshold = True
                module.target_sparsity = 1 - (1 - s) * alpha
                module.sparse_ratio = alpha

                channels = max(int(in_channel * (1 - s) * alpha), 1)
                overall_budget = in_channel * out_channel * (1 - s)
                sparse_budget = channels * out_channel
                low_rank_budget = overall_budget - sparse_budget
                module.rank = max(int(low_rank_budget / (in_channel + out_channel)), 1)
    else:
        for module_name, module in model.named_modules():
            if isinstance(module, module_type):
                if 'self_attn' in module_name:
                    in_channel = config.hidden_size
                    out_channel = config.hidden_size
                else:
                    if 'down_proj' in module_name:
                        in_channel = config.intermediate_size
                        out_channel = config.hidden_size
                    else:
                        in_channel = config.hidden_size
                        out_channel = config.intermediate_size

                module.flag_getting_threshold = True
                module.target_sparsity = 1 - (1 - args.target_sparsity) * args.sparse_ratio
                module.sparse_ratio = args.sparse_ratio

                channels = int(in_channel * (1 - args.target_sparsity) * args.sparse_ratio)
                overall_budget = in_channel * out_channel * (1 - args.target_sparsity)
                sparse_budget = channels * out_channel
                low_rank_budget = overall_budget - sparse_budget
                module.rank = int(low_rank_budget / (in_channel + out_channel))
    model._load_low_rank_module(config)

    # getting dataset
    print('Estimating threshold...')
    dataloader = get_wikitext2(nsamples=1, seed=42, seqlen=512, tokenizer=tokenizer)
    with torch.no_grad():
        inputs = torch.cat(dataloader, dim=0)
        try:
            input_device = model.model.embed_tokens.weight.device
        except Exception:
            input_device = next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cpu"))
        inputs = inputs.to(input_device)
    lm_logits = model(input_ids=inputs).logits

    for module_name, module in model.named_modules():
        if isinstance(module, module_type):
            module.prefill_ratio = args.prefill_ratio
            module.protect_prefill = not getattr(args, "disable_prefill_protection", False)
            if module.sparse_ratio == 1:
                module.mode = 'sparse'
            elif module.sparse_ratio == 0:
                module.mode = 'low_rank'
            else:
                module.mode = 'r_sparse'
    if getattr(args, "arc_saved_dir", None):
        arc_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ARCQuant", "model"))
        if arc_model_dir not in sys.path:
            sys.path.append(arc_model_dir)
        from bridge import ArcQuantBridge

        bridge = ArcQuantBridge.from_saved(
            model_name=args.model_name,
            saved_dir=args.arc_saved_dir,
            dataset=args.arc_dataset,
            metric=args.arc_metric,
            quant_type=args.arc_quant_type,
        )
        for module in model.modules():
            if isinstance(module, module_type):
                module.arc_quant_bridge = bridge
    return model

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    return trainloader

def setup_config(config, args):
    config_file = os.path.abspath(args.config_file)
    with open(config_file) as f:
        config_data = json.load(f)

    config.q_threshold = config_data['q_threshold']
    config.q_svd_path = [_resolve_relative_to(config_file, p) for p in config_data['q_svd_path']]
    config.q_low_rank = config_data['q_low_rank']

    config.k_threshold = config_data['k_threshold']
    config.k_svd_path = [_resolve_relative_to(config_file, p) for p in config_data['k_svd_path']]
    config.k_low_rank = config_data['k_low_rank']

    config.v_threshold = config_data['v_threshold']
    config.v_svd_path = [_resolve_relative_to(config_file, p) for p in config_data['v_svd_path']]
    config.v_low_rank = config_data['v_low_rank']

    config.o_threshold = config_data['o_threshold']
    config.o_svd_path = [_resolve_relative_to(config_file, p) for p in config_data['o_svd_path']]
    config.o_low_rank = config_data['o_low_rank']

    config.gate_threshold = config_data['gate_threshold']
    config.gate_svd_path = [_resolve_relative_to(config_file, p) for p in config_data['gate_svd_path']]
    config.gate_low_rank = config_data['gate_low_rank']

    config.up_threshold = config_data['up_threshold']
    config.up_svd_path = [_resolve_relative_to(config_file, p) for p in config_data['up_svd_path']]
    config.up_low_rank = config_data['up_low_rank']

    config.down_threshold = config_data['down_threshold']
    config.down_svd_path = [_resolve_relative_to(config_file, p) for p in config_data['down_svd_path']]
    config.down_low_rank = config_data['down_low_rank']

    return config
