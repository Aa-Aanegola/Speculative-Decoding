'''
Profiling functions for VLLM. (used initially when we didn't have cluster access). Full benchmarks are done with benchmarks/vllm_profile.py
'''
import torch
import wandb
from time import perf_counter
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

def vllm_profile(
    llm,
    prompts,
    sampling_params,
    wandb_name="llm_profiling",
    profile_dir="./tb_profile",
    num_warmup=3,
    wandb_entity="speculative-decoding",
    wandb_project="profiling"
):
    '''
    Profiling function for VLLM.
    Args:
        llm: VLLM instance
        prompts: list of prompts to generate from
        sampling_params: sampling parameters
        wandb_name: name of the wandb run
        profile_dir: directory to save the profile results
        num_warmup: number of warmup steps
        wandb_entity: wandb entity
        wandb_project: wandb project
    Returns:
        dict: profiling results
    '''
    wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_name)
    
    for _ in range(num_warmup):
        _ = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(profile_dir)
    ) as prof:
        with record_function("llm_generate"):
            torch.cuda.reset_peak_memory_stats()
            start_time = perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            torch.cuda.synchronize()
            end_time = perf_counter()
            peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB

    # Calculate batch-level stats
    total_tokens_generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_time = end_time - start_time
    tokens_per_second = total_tokens_generated / total_time if total_time > 0 else 0
    ms_per_token = 1000 * total_time / total_tokens_generated if total_tokens_generated > 0 else 0

    # Profiler: CPU and CUDA time (microseconds)
    cpu_time_us = sum([e.cpu_time_total for e in prof.key_averages() if hasattr(e, "cpu_time_total") and e.cpu_time_total is not None])
    cuda_time_us = sum([e.cuda_time for e in prof.key_averages() if hasattr(e, "cuda_time") and e.cuda_time is not None])
    cpu_time_ms = cpu_time_us / 1000
    cuda_time_ms = cuda_time_us / 1000
    cpu_time_per_token = cpu_time_ms / total_tokens_generated if total_tokens_generated > 0 else 0
    cuda_time_per_token = cuda_time_ms / total_tokens_generated if total_tokens_generated > 0 else 0

    print(f"Batch stats:")
    print(f"  Total tokens generated: {total_tokens_generated}")
    print(f"  Batch time: {total_time:.3f} s")
    print(f"  Tokens/sec: {tokens_per_second:.2f}")
    print(f"  ms/token (wall clock): {ms_per_token:.2f}")
    print(f"  CPU time per token: {cpu_time_per_token:.2f} ms")
    print(f"  CUDA (GPU) time per token: {cuda_time_per_token:.2f} ms")
    print(f"  Peak GPU memory: {peak_gpu_mem:.1f} MB")

    wandb.log({
        "num_prompts": len(prompts),
        "total_tokens_generated": total_tokens_generated,
        "batch_generation_time_s": total_time,
        "batch_tokens_per_second": tokens_per_second,
        "batch_ms_per_token": ms_per_token,
        "batch_peak_gpu_memory_MB": peak_gpu_mem,
        "batch_cpu_time_per_token_ms": cpu_time_per_token,
        "batch_cuda_time_per_token_ms": cuda_time_per_token,
        "batch_total_cpu_time_ms": cpu_time_ms,
        "batch_total_cuda_time_ms": cuda_time_ms,
    })

    wandb.finish()
    print(f"Profiling complete. See tensorboard logs in '{profile_dir}'.")
    return {
        "outputs": outputs,
        "total_tokens_generated": total_tokens_generated,
        "batch_time": total_time,
        "tokens_per_second": tokens_per_second,
        "ms_per_token": ms_per_token,
        "peak_gpu_mem": peak_gpu_mem,
        "cpu_time_per_token": cpu_time_per_token,
        "cuda_time_per_token": cuda_time_per_token
    }


def full_vllm_profile(
    llm,
    prompts,
    sampling_params,
    wandb_name=None,
    profile_dir="./tb_profile",
    num_warmup=3,
    batch_size=50,
    wandb_entity="speculative-decoding",
    wandb_project="profiling",
):
    '''
    Profiling function for VLLM.
    Args:
        llm: VLLM instance
        prompts: list of prompts to generate from
        sampling_params: sampling parameters
        wandb_name: name of the wandb run
        profile_dir: directory to save the profile results
        num_warmup: number of warmup steps
        batch_size: batch size
        wandb_entity: wandb entity
        wandb_project: wandb project
    Returns:
        dict: profiling results
    '''
    if wandb_name is None:
        wandb_name = "full_vllm_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_name)

    for _ in range(num_warmup):
        _ = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
    all_tokens = 0
    all_time = 0
    all_cpu_time = 0
    all_cuda_time = 0
    all_gpu_mem = 0

    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx: batch_idx + batch_size]

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=False,
            record_shapes=False,
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(profile_dir)
        ) as prof:
            with record_function("llm_generate_batch"):
                torch.cuda.reset_peak_memory_stats()
                start_time = perf_counter()
                outputs = llm.generate(batch_prompts, sampling_params)
                torch.cuda.synchronize()
                end_time = perf_counter()
                peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        batch_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        batch_time = end_time - start_time

        # Profiler stats
        cpu_time_us = sum([e.cpu_time_total for e in prof.key_averages() if hasattr(e, "cpu_time_total") and e.cpu_time_total is not None])
        cuda_time_us = sum([e.cuda_time for e in prof.key_averages() if hasattr(e, "cuda_time") and e.cuda_time is not None])
        cpu_time_ms = cpu_time_us / 1000
        cuda_time_ms = cuda_time_us / 1000
        cpu_time_per_token = cpu_time_ms / batch_tokens if batch_tokens > 0 else 0
        cuda_time_per_token = cuda_time_ms / batch_tokens if batch_tokens > 0 else 0
        tokens_per_second = batch_tokens / batch_time if batch_time > 0 else 0
        ms_per_token = 1000 * batch_time / batch_tokens if batch_tokens > 0 else 0

        # Log to wandb (trend over time: one point per batch)
        wandb.log({
            "batch_idx": batch_idx // batch_size,
            "num_prompts": len(batch_prompts),
            "batch_tokens": batch_tokens,
            "batch_generation_time_s": batch_time,
            "batch_tokens_per_second": tokens_per_second,
            "batch_ms_per_token": ms_per_token,
            "batch_peak_gpu_memory_MB": peak_gpu_mem,
            "batch_cpu_time_per_token_ms": cpu_time_per_token,
            "batch_cuda_time_per_token_ms": cuda_time_per_token,
            "batch_total_cpu_time_ms": cpu_time_ms,
            "batch_total_cuda_time_ms": cuda_time_ms,
        })

        # Aggregate totals
        all_tokens += batch_tokens
        all_time += batch_time
        all_cpu_time += cpu_time_ms
        all_cuda_time += cuda_time_ms
        all_gpu_mem = max(all_gpu_mem, peak_gpu_mem)

    # Log final summary
    final_tokens_per_second = all_tokens / all_time if all_time > 0 else 0
    final_ms_per_token = 1000 * all_time / all_tokens if all_tokens > 0 else 0
    final_cpu_time_per_token = all_cpu_time / all_tokens if all_tokens > 0 else 0
    final_cuda_time_per_token = all_cuda_time / all_tokens if all_tokens > 0 else 0

    print("\nFinal Results:")
    print(f"  Total tokens generated: {all_tokens}")
    print(f"  Total time: {all_time:.3f} s")
    print(f"  Tokens/sec: {final_tokens_per_second:.2f}")
    print(f"  ms/token (wall clock): {final_ms_per_token:.2f}")
    print(f"  CPU time per token: {final_cpu_time_per_token:.2f} ms")
    print(f"  CUDA (GPU) time per token: {final_cuda_time_per_token:.2f} ms")
    print(f"  Peak GPU memory: {all_gpu_mem:.1f} MB")

    wandb.log({
        "FINAL_total_tokens": all_tokens,
        "FINAL_total_time_s": all_time,
        "FINAL_tokens_per_second": final_tokens_per_second,
        "FINAL_ms_per_token": final_ms_per_token,
        "FINAL_total_cpu_time_ms": all_cpu_time,
        "FINAL_total_cuda_time_ms": all_cuda_time,
        "FINAL_cpu_time_per_token_ms": final_cpu_time_per_token,
        "FINAL_cuda_time_per_token_ms": final_cuda_time_per_token,
        "FINAL_peak_gpu_memory_MB": all_gpu_mem,
    })

    wandb.finish()
    print("Profiling complete! Check your wandb dashboard for the trends and final values.")
    return {
        "total_tokens": all_tokens,
        "total_time": all_time,
        "tokens_per_second": final_tokens_per_second,
        "ms_per_token": final_ms_per_token,
        "cpu_time_per_token": final_cpu_time_per_token,
        "cuda_time_per_token": final_cuda_time_per_token,
        "peak_gpu_mem": all_gpu_mem,
    }

