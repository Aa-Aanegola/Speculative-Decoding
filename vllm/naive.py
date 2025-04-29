# Single-draft, fixed-length speculative decoding with rejection sampling.

import argparse
import json
from datetime import datetime
from profilefuncs import full_vllm_profile, vllm_profile
from vllm import LLM, SamplingParams
from config import PROMPTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "mini"], required=True, help="Choose profiling mode: full dataset or mini prompts")
    args = parser.parse_args()

    llm = LLM(
        model="facebook/opt-2.7b",
        tensor_parallel_size=1,
        speculative_config={
            "model": "facebook/opt-125m",
            "num_speculative_tokens": 5,
        },
    )

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "full":
        with open("../benchmarks/data/question.jsonl") as f:
            prompts = [json.loads(line)["turns"][0] for line in f]

        name = f"full_vllm_naive_opt2.7b_opt125m_{timestamp}"
        profile_dir = f"./tbprofiles/{name}"

        full_vllm_profile(
            llm,
            prompts,
            sampling_params,
            wandb_name=name,
            profile_dir=profile_dir,
            batch_size=50,
            num_warmup=3
        )

    elif args.mode == "mini":
        prompts = PROMPTS
        name = f"vllm_naive_opt2.7b_opt125m_{timestamp}"
        profile_dir = f"./tbprofiles/{name}"

        vllm_profile(
            llm,
            prompts,
            sampling_params,
            wandb_name=name,
            profile_dir=profile_dir,
            num_warmup=3
        )

if __name__ == "__main__":
    main()
