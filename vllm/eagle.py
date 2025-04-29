
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
        model="Qwen/Qwen2-7B-Instruct",  # Main model (7B)
        tensor_parallel_size=1,
        max_model_len=16320,
        enforce_eager=True,
        speculative_config={
            "model": "yuhuili/EAGLE-Qwen2-7B-Instruct",  # Draft model (EAGLE, small)
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": 5,
        },
    )
    # llm = LLM(
    #     model="lmsys/vicuna-7b-v1.3",  # Main model (7B)
    #     tensor_parallel_size=1,
    #     # max_model_len=16320,
    #     enforce_eager=True,
    #     speculative_config={
    #         "model": "yuhuili/EAGLE-Vicuna-7B-v1.3",  # Draft model (EAGLE, small)
    #         "draft_tensor_parallel_size": 1,
    #         "num_speculative_tokens": 5,
    #     },
    # )
    # llm = LLM(
    #     model="meta-llama/Meta-Llama-3-8B-Instruct",  # Main model (7B)
    #     tensor_parallel_size=1,
    #     # max_model_len=16320,
    #     # enforce_eager=True,
    #     speculative_config={
    #         "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",  # Draft model (EAGLE, small)
    #         "draft_tensor_parallel_size": 1,
    #         "num_speculative_tokens": 5,
    #     },
    # )

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "full":
        with open("../benchmarks/data/question.jsonl") as f:
            prompts = [json.loads(line)["turns"][0] for line in f]

        name = f"full_vllm_eagle_qwen7b_eager_{timestamp}"
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
        name = f"vllm_eagle_qwen7b_eager_{timestamp}"
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
