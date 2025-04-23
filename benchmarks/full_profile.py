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
os.environ["HF_HOME"] = f"/insomnia001/depts/edu/COMSE6998/{UNI}/.cache/huggingface"
os.environ["WANDB_DIR"] = f"/insomnia001/depts/edu/COMSE6998/{UNI}/speculative-decoding/benchmarks/"
os.environ["WANDB_CACHE_DIR"] = f"/insomnia001/depts/edu/COMSE6998/{UNI}/.cache/wandb"

def log_to_wandb(entry_result, question_id, turn_index, category):
    wandb.log({
        "question_id": question_id,
        "turn_index": turn_index,
        "category": category,
        "num_tokens": len(entry_result["tokens"]),
        "total_time": entry_result["total_time"],
        "tokens_per_second": entry_result["tokens_per_second"]
    })

@torch.no_grad()
def profile_single_turn(model, tokenizer, prompt: str, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    start = now()

    ## TODO: Figure out how to do token by token generation

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    torch.cuda.synchronize()
    end = now()

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    total_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
    total_time = end - start

    return {
        "output": decoded,
        "total_time": total_time,
        "tokens": decoded,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
    }


def profile_dataset(model, tokenizer, dataset: List[Dict], max_new_tokens=50):
    results = []

    for entry in dataset:
        qid = entry["question_id"]
        category = entry.get("category", "unknown")
        for i, turn in enumerate(entry["turns"]):
            result = profile_single_turn(model, tokenizer, turn, max_new_tokens=max_new_tokens)

            results.append(result)

            print(f"[{qid} - Turn {i}] {len(result['tokens'])} tokens, {result['total_time']:.3f}s total, {result['tokens_per_second']:.2f} tok/s")
            log_to_wandb(result, qid, i, category)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    parser.add_argument('--dataset', type=str, required=True, help='dataset to profile')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
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
    
    
    model, tokenizer, args = load_model(config)
    generate_args = {**args, **config["generate_args"]}
    
    # Log model statistics
    wandb.log({
        "model_name": str(model),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers": len(list(model.children()))
    })
    
    res = profile_dataset(model, tokenizer, dset, max_new_tokens=generate_args["max_new_tokens"])
    with open(f'/insomnia001/depts/edu/COMSE6998/{UNI}/results_{config["type"]}.json', 'w') as f:
        json.dump(res, f, indent=4)
    wandb.finish()
    print("Profiling complete.")