import sys
sys.path.append('../EAGLE')
sys.path.append('../')
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
        
# Helper function to write results to wandb
def log_to_wandb(entry_result, _id, cluster):
    wandb.log({
        "id": _id,
        "cluster": cluster,
        "num_tokens": len(entry_result["output"]),
        "total_time": entry_result["total_time"],
        "tokens_per_second": entry_result["tokens_per_second"]
    })

# Run profile for a single prompt 
@torch.no_grad()
def profile_single_turn(model, tokenizer, generate_args, prompt: str, max_new_tokens=50):
    # Special parsing for Eagle - it uses the conversation template
    if model.__class__.__name__ == "EaModel":
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("llama3")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    torch.cuda.synchronize()
    start = now()

    # Special handling for eagle, it doesn't take all inputs but only IDs
    if model.__class__.__name__ == "EaModel":
        output = model.eagenerate(
            inputs.input_ids, 
            **generate_args
        )
    else:
        output = model.generate(
            **inputs,
            **generate_args
        )

    torch.cuda.synchronize()
    end = now()

    # Post processing and decoding 
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
        try:
            result = profile_single_turn(model, tokenizer, generate_args, prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            # some prompts break some of the models - explicit error handling
            print(f"[{_id}] Error: {e}")
            result = {
                "output": "",
                "total_time": 0,
                "tokens_per_second": 0,
            }
    
        results.append({
            **result,
            **entry
        })

        print(f"[{_id}] {len(result['output'])} tokens, {result['total_time']:.3f}s total, {result['tokens_per_second']:.2f} tok/s")
        log_to_wandb(result, _id, entry['cluster'])

    return results


if __name__ == "__main__":
    # Argument parsing 
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    parser.add_argument('--dataset', type=str, required=True, help='dataset to profile')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--output-dir', type=str, default='../out', help='directory to save results')
    args = parser.parse_args()
    
    with open(f'{args.yaml}', 'r') as f:
        config = yaml.safe_load(f)

    dset = load_jsonl(args.dataset)

    if args.wandb:
        os.environ["WANDB_DISABLED"] = "false"
    else:
        os.environ["WANDB_DISABLED"] = "true"
    
    # WandB setup
    run = wandb.init(
        entity="speculative-decoding",
        project="full-profiles",
        config=config,
        name=config["type"]+ '_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    
    # Model loading based on config 
    model, tokenizer, model_args = load_model(config)
    generate_args = {**model_args, **config["generate_args"]}
    
    wandb.log({
        "model_name": str(model),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers": len(list(model.children()))
    })
    
    # Profile the dataset
    res = profile_dataset(model, tokenizer, generate_args, dset, max_new_tokens=generate_args["max_new_tokens"])
    results_file = os.path.join(args.output_dir, f'results_{config["type"]}.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Save the results 
    with open(results_file, 'w') as f:
        json.dump(res, f, indent=4)
    wandb.finish()
    print(f"Profiling complete. Results saved to {results_file}")