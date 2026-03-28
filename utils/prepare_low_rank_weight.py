import os 
import torch
import argparse
import torch.nn as nn
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            output_path = os.path.join(args.output_dir, name + '.pt')
            if args.skip_existing and os.path.exists(output_path):
                print(f"Skipping existing low-rank factors: {name}")
                continue

            u, s, v = torch.svd(m.weight.to(torch.float64).cuda())
            weight_reconstructed = u @ torch.diag(s) @ v.T

            u = u.to(torch.float16)
            s = s.to(torch.float16)
            v = v.to(torch.float16)

            scale = v @ torch.diag(s)
            scale = scale.norm(dim=1)

            error = torch.norm(m.weight.cuda() - weight_reconstructed)
            torch.save((u.cpu(), s.cpu(), v.cpu(), scale.cpu()), output_path)
            print(f"Saved low-rank factors: {name} | reconstruction_residual={error.item():.6e}")

if __name__ == '__main__':
    main()
