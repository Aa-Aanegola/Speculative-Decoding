import torch
from typing import List, Dict
import yaml
from utils import load_model, load_jsonl
from data import *
from tqdm import tqdm
import wandb
from datetime import datetime
import os
from time import perf_counter as now

# Use scratch directory for cache and output
SCRATCH_DIR = "/insomnia001/depts/edu/COMSE6998/nb3227/scratch"
os.makedirs(SCRATCH_DIR, exist_ok=True)

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
def profile_single_turn(model, tokenizer, generate_args, prompt: str):
    # Tokenize + move to device, with truncation
    max_input_length = 1024  # Leave room for generation
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    ).to(model.device)

    torch.cuda.synchronize()
    start = now()

    # ——— Medusa branch ———
    if hasattr(model, 'medusa_generate'):
        # Reset KV cache if it exists
        if hasattr(model, 'past_key_values'):
            model.past_key_values = None
            if hasattr(model, 'current_length_data'):
                model.current_length_data.zero_()

        # Extract only the args medusa_generate declares
        medusa_args = {
            k: generate_args[k]
            for k in [
                'temperature', 'max_steps', 'medusa_choices',
                'posterior_threshold', 'posterior_alpha',
                'top_p', 'sampling', 'fast'
            ]
            if k in generate_args
        }

        # Streaming generation
        output_text = ""
        for chunk in model.medusa_generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            **medusa_args
        ):
            output_text = chunk["text"]

        torch.cuda.synchronize()
        end = now()

        total_time = end - start
        total_tokens = len(tokenizer.encode(output_text))

        return {
            "output": output_text,
            "total_time": total_time,
            "tokens": output_text,
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        }

    # ——— Standard HF‐style branch ———
    else:
        output = model.generate(
            **inputs,
            **generate_args
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

def profile_dataset(model, tokenizer, generate_args, dataset: List[Dict]):
    results = []
    for entry in dataset:
        qid = entry["question_id"]
        category = entry.get("category", "unknown")
        for i, turn in enumerate(entry["turns"]):
            result = profile_single_turn(model, tokenizer, generate_args, turn, max_new_tokens=max_new_tokens)

            results.append({
                **result,
                **entry
            })

            print(f"[{qid}] {len(result['tokens'])} tokens, {result['total_time']:.3f}s total, {result['tokens_per_second']:.2f} tok/s")
            log_to_wandb(result, qid, i, category)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml',      type=str,  required=True)
    parser.add_argument('--dataset',   type=str,  required=True)
    parser.add_argument('--wandb',     action='store_true')
    parser.add_argument('--output-dir',type=str,  default=SCRATCH_DIR)
    args = parser.parse_args()

    # Load configs & data
    config = yaml.safe_load(open(args.yaml))
    dset   = load_jsonl(args.dataset)

    # Toggle wandb
    os.environ["WANDB_DISABLED"] = "false" if args.wandb else "true"
    run = wandb.init(
        entity="speculative-decoding",
        project="full-profiles",
        config=config,
        name=f"{config['type']}_{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )
    
    # Load model + tokenizer + model_args
    model, tokenizer, model_args = load_model(config)

    # **MEDUSA**: use model_args directly (they already match medusa_generate)
    if config["type"] == "medusa":
        generate_args = model_args
    else:
        # other types: merge your HF‐style generate_args
        generate_args = {**model_args, **config["generate_args"]}
    
    # Log some stats
    wandb.log({
        "model_name":    str(model),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers":    len(list(model.children()))
    })
    
    # Run the profiling
    res = profile_dataset(model, tokenizer, generate_args, dset)

    # Save & finish
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"results_{config['type']}.json")
    with open(out_path, "w") as f:
        import json
        json.dump(res, f, indent=2)
    wandb.finish()
    print(f"Profiling complete. Results saved to {out_path}")
