from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import torch
from typing import List, Dict
import yaml
from utils import *
from data import *
import os
from tqdm import tqdm
import wandb
from datetime import datetime


# wrapper function to save pytorch profiler data
def save_profile(model, tokenizer, generate_args: Dict, prompt_list: List[str], log_dir: str, wandb_run):    
    log_dir = f'{logdir_base}/{log_dir}/'
    os.makedirs(log_dir, exist_ok=True)

    # Warm-up runs
    for i in range(3):
        inputs = tokenizer(prompt_list[i], return_tensors='pt').to(device)
        _ = model.generate(**inputs, **generate_args)

    # Profiling a single prompt
    prompt = prompt_list[3]
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(log_dir)
    ) as prof:
        with record_function("model_generate"):
            outputs = model.generate(**inputs, **generate_args)

    num_tokens = outputs.shape[-1]
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    key_averages = prof.key_averages()
    total_cuda_time_ms = sum([getattr(item, 'cuda_time', 0) for item in key_averages]) / 1e3
    total_cpu_time_ms = sum([getattr(item, 'cpu_time', 0) for item in key_averages]) / 1e3
    max_gpu_mem = torch.cuda.max_memory_allocated() / 1e6

    num_tokens = outputs.shape[-1]

    wandb_run.log({
        "tokens_generated": num_tokens,
        "total_cpu_time_ms": total_cpu_time_ms,
        "cpu_time_per_token_ms": total_cpu_time_ms / num_tokens,
        "total_cuda_time_ms": total_cuda_time_ms,
        "cuda_time_per_token_ms": total_cuda_time_ms / num_tokens,
        "max_gpu_memory_MB": max_gpu_mem,
        "gpu_mem_per_token_MB": max_gpu_mem / num_tokens,
    })


    stats = prof.key_averages().table(sort_by="cuda_time", row_limit=10)
    for row in stats:
        wandb_run.log({
            f"{row.key}_cuda_time_per_token_ms": (row.self_cuda_time_total / 1e3) / num_tokens,
            f"{row.key}_cpu_time_per_token_ms": (row.self_cpu_time_total / 1e3) / num_tokens,
        })

    op_table = wandb.Table(columns=["Op", "CUDA Time/token (ms)", "CPU Time/token (ms)"])
    for row in prof.key_averages():
        op_table.add_data(
            row.key,
            (row.cuda_time / 1e3) / num_tokens,
            (row.cpu_time / 1e3) / num_tokens,
        )
    wandb_run.log({"profiler_summary": op_table})
    wandb_run.log({"tensorboard_trace_path": log_dir})




if __name__ == "__main__":
    # load specified yaml file in command line args
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    args = parser.parse_args()
    with open(f'{args.yaml}', 'r') as f:
        config = yaml.safe_load(f)
    
    run = wandb.init(
        entity="speculative-decoding",
        project="profiling",
        config=config,
        name=config["type"]+ '_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    
    
    target_model, tokenizer, args = load_model(config)
    prompts = get_data()
    generate_args = {**args, **config["generate_args"]}
    
    # Log model statistics
    wandb.log({
        "model_name": str(target_model),
        "num_parameters": sum(p.numel() for p in target_model.parameters()),
        "num_layers": len(list(target_model.children()))
    })
    
    save_profile(target_model, tokenizer, generate_args, prompts, config["type"], run)