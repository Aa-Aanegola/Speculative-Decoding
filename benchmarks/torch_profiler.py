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
os.environ["HF_HOME"] = f"/insomnia001/depts/edu/COMSE6998/{UNI}/.cache/huggingface"
os.environ["WANDB_DIR"] = f"/insomnia001/depts/edu/COMSE6998/{UNI}/speculative-decoding/benchmarks/"


# wrapper function to save pytorch profiler data
def save_profile(model, tokenizer, generate_args: Dict, prompt_list: List[str], log_dir: str, wandb_run):    
    log_dir = f'/insomnia001/depts/edu/COMSE6998/{UNI}/{logdir_base}/{log_dir}/'
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
    total_cuda_time_ms = sum([getattr(item, 'device_time', 0) for item in key_averages]) / 1e3
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

    op_table = wandb.Table(columns=["Op", "CUDA Time/token (ms)", "CPU Time/token (ms)"])
    for row in prof.key_averages():
        op_table.add_data(
            row.key,
            (row.device_time / 1e3) / num_tokens,
            (row.cpu_time / 1e3) / num_tokens,
        )
    wandb_run.log({"profiler_summary": op_table})
    wandb_run.log({"tensorboard_trace_path": log_dir})




if __name__ == "__main__":
    # load specified yaml file in command line args
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    args = parser.parse_args()
    with open(f'{args.yaml}', 'r') as f:
        config = yaml.safe_load(f)

    if args.wandb:
        os.environ["WANDB_DISABLED"] = "false"
    else:
        os.environ["WANDB_DISABLED"] = "true"
    
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