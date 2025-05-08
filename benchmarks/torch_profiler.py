import sys
sys.path.append('../')

from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import torch
from typing import List, Dict
import yaml
from utils import *
import os
from tqdm import tqdm
import wandb
from datetime import datetime
import logging
os.environ["WANDB_MODE"] = "disabled"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append('../../UmbreLLa')

# wrapper function to save pytorch profiler data
def save_profile(model, tokenizer, generate_args: Dict, prompt_list: List[str], log_dir: str, wandb_run):    
    # Use absolute path
    base_dir = os.path.abspath(os.path.dirname(__file__))
    full_log_dir = os.path.join(base_dir, logdir_base, log_dir)
    logger.info(f"Using log directory: {full_log_dir}")
    
    # Ensure the directory exists and is empty
    os.makedirs(full_log_dir, exist_ok=True)
    logger.info(f"Created directory: {full_log_dir}")
    
    # Clean any existing files
    for f in os.listdir(full_log_dir):
        file_path = os.path.join(full_log_dir, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            logger.error(f"Error cleaning directory: {e}")

    # Warm-up runs
    for i in range(3):
        inputs = tokenizer(prompt_list[i], return_tensors='pt').to(device)
        _ = model.generate(**inputs, **generate_args)

    # Profiling a single prompt
    prompt = prompt_list[3]
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Create a unique subdirectory for this profiling run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_dir = os.path.join(full_log_dir, f"run_{timestamp}")
    os.makedirs(profile_dir, exist_ok=True)
    logger.info(f"Created profile directory: {profile_dir}")

    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(profile_dir)
        ) as prof:
            with record_function("model_generate"):
                outputs = model.generate(**inputs, **generate_args)
    except Exception as e:
        logger.error(f"Error during profiling: {e}")
        raise

    try:
        num_tokens = outputs.shape[-1]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        key_averages = prof.key_averages()
        total_cuda_time_ms = sum([getattr(item, 'device_time', 0) for item in key_averages]) / 1e3
        total_cpu_time_ms = sum([getattr(item, 'cpu_time', 0) for item in key_averages]) / 1e3
        max_gpu_mem = torch.cuda.max_memory_allocated() / 1e6

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
        wandb_run.log({"tensorboard_trace_path": profile_dir})
        logger.info("Successfully completed profiling and logging")
    except Exception as e:
        logger.error(f"Error processing profiling results: {e}")
        raise




if __name__ == "__main__":
    # load specified yaml file in command line args
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--dataset', type=str, required=True, help='dataset to profile')
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
    prompts = [item['question'] for item in load_jsonl(args.dataset)]
    generate_args = {**args, **config["generate_args"]}
    
    # Log model statistics
    wandb.log({
        "model_name": str(target_model),
        "num_parameters": sum(p.numel() for p in target_model.parameters()),
        "num_layers": len(list(target_model.children()))
    })
    
    save_profile(target_model, tokenizer, generate_args, prompts, config["type"], run)