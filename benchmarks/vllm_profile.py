import time
import json
import os
import argparse
import yaml
from typing import List, Dict
from datetime import datetime
from vllm import LLM, SamplingParams
import wandb

def log_to_wandb(entry_result: Dict):
    """
    Logs individual prompt profiling results to Wandb.

    Args:
        entry_result: Dictionary containing results for a single prompt.
    """
    _id = entry_result.get("id", "N/A")
    cluster = entry_result.get("cluster", "N/A")
    num_tokens = entry_result.get("num_tokens", 0)
    total_time = entry_result.get("total_time", 0)
    tokens_per_second = entry_result.get("tokens_per_second", 0)
    error = entry_result.get("error", None)

    log_data = {
        "id": _id,
        "cluster": cluster,
        "num_tokens": num_tokens,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second,
    }

    if error:
        log_data["error"] = error

    try:
        wandb.log(log_data)
    except Exception as e:
        print(f"Warning: Could not log data for ID {_id} to wandb: {e}")


def profile_vllm_dataset(llm: LLM, sampling_params: SamplingParams, dataset: List[Dict], max_new_tokens: int):
    """
    Profiles text generation speed for a dataset of prompts using vLLM.

    Args:
        llm: The initialized vLLM LLM object.
        sampling_params: The SamplingParams object for generation.
        dataset: A list of dictionaries, where each dict contains at least
                 "id" and "prompt".
        max_new_tokens: The maximum number of new tokens to generate per prompt.
    """
    results = []

    print(f"\n--- Starting Profiling for {len(dataset)} prompts ---")

    for i, entry in enumerate(dataset):
        _id = entry.get("id", f"prompt_{i:04d}")
        prompt = entry["prompt"]

        entry_results: Dict = {}

        try:
            # Start timing
            start_time = time.perf_counter()

            outputs = llm.generate([prompt], sampling_params)
            # no need for cuda sync here

            # Stop timing
            end_time = time.perf_counter()

            if not outputs or not outputs[0].outputs:
                 raise ValueError("vLLM generation failed or returned no output.")

            output = outputs[0] 
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids

            total_time = end_time - start_time
            num_tokens = len(token_ids)
            tokens_per_second = num_tokens / total_time if total_time > 0 else 0

            print(f"[{_id}] Generated {num_tokens} tokens in {total_time:.3f}s ({tokens_per_second:.2f} tok/s)")

            entry_results = {
                **entry,
                "id": _id,
                "generated_text": generated_text,
                "num_tokens": num_tokens,
                "total_time": total_time,
                "tokens_per_second": tokens_per_second,
            }


        except Exception as e:
            print(f"[{_id}] Error during generation: {e}")
            entry_results = {
                **entry,
                "id": _id,
                "generated_text": "",
                "num_tokens": 0,
                "total_time": 0,
                "tokens_per_second": 0,
                "error": str(e)
            }

        log_to_wandb(entry_results)

        results.append(entry_results)


    return results

def run_warmup(llm: LLM, sampling_params: SamplingParams, num_warmup_prompts: int = 5, max_warmup_tokens: int = 30):
    """
    Runs a few dummy generations to warm up the vLLM model.
    """
    print(f"\n--- Running Warmup ({num_warmup_prompts} prompts, max {max_warmup_tokens} tokens each) ---")
    if num_warmup_prompts <= 0:
        print("Skipping warmup as num_warmup_prompts is zero or negative.")
        return

    warmup_prompts = [f"This is a warmup prompt number {i} to load kernels and memory." for i in range(num_warmup_prompts)]

    warmup_sampling_params = SamplingParams(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=max_warmup_tokens,
    )

    try:
        llm.generate(warmup_prompts, warmup_sampling_params, use_tqdm=False)
        print("Warmup complete.")
    except Exception as e:
        print(f"Warning: Warmup failed with error: {e}")
        print("Continuing without successful warmup. Profiling results might be less accurate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vLLM Profiler with Warmup and Wandb')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load vLLM and sampling config')
    parser.add_argument('--dataset', type=str, required=True, help='dataset file (JSONL format) with prompts')
    parser.add_argument('--output-dir', type=str, default='./vllm_out', help='directory to save results')
    parser.add_argument('--num-warmup-prompts', type=int, default=5, help='Number of dummy prompts for warmup')
    parser.add_argument('--max-warmup-tokens', type=int, default=30, help='Max tokens for warmup generations')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-entity', type=str, default="speculative-decoding", help='Wandb entity name')
    parser.add_argument('--wandb-project', type=str, default="vllm-routing", help='Wandb project name')

    args = parser.parse_args()

    # Load configuration from YAML
    try:
        with open(args.yaml, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.yaml}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        exit(1)

    # Define vLLM specific configs
    vllm_model_config = config.get("vllm_model", {})
    sampling_config = config.get("sampling_params", {})
    generate_config = config.get("generate_args", {})

    model_name = vllm_model_config.get("model", "meta-llama/Llama-3.1-8B-Instruct")

    # Configure and Initialize Wandb 
    if args.wandb:
        os.environ["WANDB_DISABLED"] = "false"
        print("Wandb logging enabled.")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("Wandb logging disabled.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_model_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    spectype = args.yaml.split("/")[-1].split(".")[0]
    run_name = f"{spectype}_{clean_model_name}_{timestamp}"

    run = None

    try:
        if not os.environ.get("WANDB_DISABLED") == "true":
            run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                config=config,
                name=run_name,
                reinit=True
            )
            print(f"Wandb run initialized: {run.url}")

            wandb.log({
                "vllm_model_config": vllm_model_config,
                "sampling_config": sampling_config,
                "generate_config": generate_config,
                "dataset_file": args.dataset,
                "num_warmup_prompts": args.num_warmup_prompts,
                "max_warmup_tokens": args.max_warmup_tokens,
                "script_args": vars(args)
            })
        else:
             print("Wandb explicitly disabled by environment variable or argument.")

    except Exception as e:
         print(f"Error initializing Wandb: {e}")
         if args.wandb:
             print("Wandb initialization failed. Exiting as --wandb was set.")
             exit(1)
         else:
             print("Wandb initialization failed, but logging was not requested. Continuing without wandb.")
             os.environ["WANDB_DISABLED"] = "true"


    # Load the dataset
    dataset: List[Dict] = []
    try:
        with open(args.dataset, 'r') as f:
            for line in f:
                if line.strip():
                   dataset.append(json.loads(line))
        print(f"Loaded {len(dataset)} prompts from {args.dataset}")
        if not dataset:
             print("Warning: Dataset is empty.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset}")
        if run: run.finish(exit_code=1)
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSONL dataset file: {e}")
        if run: run.finish(exit_code=1)
        exit(1)
    except Exception as e:
        print(f"An error occurred loading the dataset: {e}")
        if run: run.finish(exit_code=1)
        exit(1)

    vllm_model_config = config.get("vllm_model", {})
    vllm_init_args = {k: v for k, v in vllm_model_config.items() if k not in ["model"]}


    # Initialize vLLM
    print(f"Initializing vLLM model: {model_name}...")
    try:
        llm = LLM(
            model=model_name,
            **vllm_init_args
        )
    except Exception as e:
         print(f"Error initializing vLLM: {e}")
         if run: run.finish(exit_code=1)
         exit(1)
    print("vLLM initialized.")

    max_new_tokens = generate_config.get("max_new_tokens", 50)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=sampling_config.get("temperature", 0.8),
        top_p=sampling_config.get("top_p", 0.95),
        **{k: v for k, v in sampling_config.items() if k not in ["temperature", "top_p"]}
    )


    # Run Warmup
    run_warmup(llm, sampling_params, args.num_warmup_prompts, args.max_warmup_tokens)

    # Run Profiling
    profile_results = profile_vllm_dataset(llm, sampling_params, dataset, max_new_tokens)

    results_file = os.path.join(args.output_dir, f'{run_name}.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    try:
        with open(results_file, 'w') as f:
            json.dump(profile_results, f, indent=4)
        print(f"\nProfiling complete. Results saved to {results_file}")
    except Exception as e:
        print(f"\nError saving results to {results_file}: {e}")

    if run:
        try:
            run.finish()
            print("Wandb run finished.")
        except Exception as e:
            print(f"Error finishing Wandb run: {e}")