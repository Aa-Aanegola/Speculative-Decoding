from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import torch
from typing import List, Dict
import yaml
from utils import *
from data import *
import os
from tqdm import tqdm



# wrapper function to save pytorch profiler data
def save_profile(model, tokenizer, generate_args: Dict, prompt_list: List[str], log_dir: str):
    log_dir = f'{logdir_base}/{log_dir}/'
    os.makedirs(log_dir, exist_ok=True)

    for i in range(3):
        inputs = tokenizer(prompt_list[i], return_tensors='pt').to(device)
        outputs = model.generate(**inputs, **generate_args)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(log_dir)
    ) as prof:
            with record_function("model_generate"):
                prompt = prompt_list[3]
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                outputs = model.generate(**inputs, **generate_args)
                tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    # load specified yaml file in command line args
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    args = parser.parse_args()
    with open(f'{args.yaml}', 'r') as f:
        config = yaml.safe_load(f)
    
    # load model and tokenizer
    target_model, tokenizer, args = load_model(config)
    prompts = get_data()
    generate_args = {**args, **config["generate_args"]}
    
    save_profile(target_model, tokenizer, generate_args, prompts, config["type"])