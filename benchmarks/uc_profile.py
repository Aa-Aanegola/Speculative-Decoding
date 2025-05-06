import torch
from typing import List, Dict
import yaml
from utils import *
from data import *
from tqdm import tqdm
import wandb
from datetime import datetime
import os
from time import perf_counter as now

# # Use scratch directory for cache and output
SCRATCH_DIR = "/insomnia001/depts/edu/COMSE6998/aa5506/scratch"
# os.environ["HF_HOME"] = os.path.join(SCRATCH_DIR, ".cache/huggingface")
# os.environ["WANDB_DIR"] = os.path.join(SCRATCH_DIR, "wandb")
# os.environ["WANDB_CACHE_DIR"] = os.path.join(SCRATCH_DIR, ".cache/wandb")

# # Create directories if they don't exist
os.makedirs(SCRATCH_DIR, exist_ok=True)
# os.makedirs(os.environ["HF_HOME"], exist_ok=True)
# os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
# os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)

def log_to_wandb(entry_result, _id, cluster):
    wandb.log({
        "id": _id,
        "cluster": cluster,
        "num_tokens": len(entry_result["output"]),
        "total_time": entry_result["total_time"],
        "tokens_per_second": entry_result["tokens_per_second"]
    })

@torch.no_grad()
def profile_single_turn(model, tokenizer, generate_args, prompt: str, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    start = now()

    ## TODO: Figure out how to do token by token generation

    output = model.generate(
        **inputs,
        **generate_args
    )

    torch.cuda.synchronize()
    end = now()

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
    total_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
    total_time = end - start

    return {
        "output": decoded,
        "total_time": total_time,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
    }


def profile_dataset(model, tokenizer, generate_args, dataset: List[Dict], max_new_tokens=50):
    results = []

    for entry in dataset:
        _id = entry["id"]
        
        prompt = entry["prompt"]
        result = profile_single_turn(model, tokenizer, generate_args, prompt, max_new_tokens=max_new_tokens)

        results.append({
            **result,
            **entry
        })

        print(f"[{_id}] {len(result['output'])} tokens, {result['total_time']:.3f}s total, {result['tokens_per_second']:.2f} tok/s")
        log_to_wandb(result, _id, entry['cluster'])

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    parser.add_argument('--dataset', type=str, required=True, help='dataset to profile')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--output-dir', type=str, default=SCRATCH_DIR, help='directory to save results')
    args = parser.parse_args()
    
    with open(f'{args.yaml}', 'r') as f:
        config = yaml.safe_load(f)

    dset = load_jsonl(args.dataset)

    if args.wandb:
        os.environ["WANDB_DISABLED"] = "false"
    else:
        os.environ["WANDB_DISABLED"] = "true"
    
    run = wandb.init(
        entity="speculative-decoding",
        project="full-profiles",
        config=config,
        name=config["type"]+ '_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    
    model, tokenizer, model_args = load_model(config)
    generate_args = {**model_args, **config["generate_args"]}
    
    # Log model statistics
    wandb.log({
        "model_name": str(model),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers": len(list(model.children()))
    })
    
    res = profile_dataset(model, tokenizer, generate_args, dset, max_new_tokens=generate_args["max_new_tokens"])
    results_file = os.path.join(args.output_dir, f'results_{config["type"]}.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(res, f, indent=4)
    wandb.finish()
    print(f"Profiling complete. Results saved to {results_file}")