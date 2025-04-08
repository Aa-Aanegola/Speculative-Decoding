import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import os

# Model identifiers
target_cpt = "google/gemma-2b"
assistant_cpt = "double7/vicuna-68m"

# Load tokenizers and models
target_tokenizer = AutoTokenizer.from_pretrained(target_cpt)
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_cpt)

target_model = AutoModelForCausalLM.from_pretrained(target_cpt)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_cpt)

# Prompt list
prompt_list = [
    'What is the capital of France?',
    'Who wrote \'1984\'?',
    'The quick brown fox jumps over the lazy...',
    'Write a Python function to compute the Fibonacci sequence.',
    'Define Newton\'s second law of motion.',
    'List three primary colors.',
    'Convert 100°F to Celsius.',
    'What are the three branches of the U.S. government?',
    'How do you define a class in Python?',
    'What\'s 8 times 7?'
]

# Ensure models are in eval mode
target_model.eval()
assistant_model.eval()

# Set to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_model.to(device)
assistant_model.to(device)

# Create log directories
logdir_target = "./logdir/target_only"
logdir_assisted = "./logdir/target_assisted"

os.makedirs(logdir_target, exist_ok=True)
os.makedirs(logdir_assisted, exist_ok=True)

# --- Target Only ---
print("\nProfiling Target Model Only...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
    on_trace_ready=tensorboard_trace_handler(logdir_target),
    with_flops=True,
    profile_memory=True
) as prof_target:
    for prompt in prompt_list:
        with record_function("target_model_only"):
            inputs = target_tokenizer(prompt, return_tensors='pt').to(device)
            outputs = target_model.generate(**inputs)
            target_tokenizer.batch_decode(outputs, skip_special_tokens=True)

print("Target-only profiling data saved to:", logdir_target)


# --- Target + Assistant ---
print("\nProfiling Target + Assistant Model...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
    on_trace_ready=tensorboard_trace_handler(logdir_assisted),
    with_flops=True,
    profile_memory=True
) as prof_assisted:
    for prompt in prompt_list:
        with record_function("target_assisted_by_vicuna"):
            inputs = target_tokenizer(prompt, return_tensors='pt').to(device)
            outputs = target_model.generate(
                **inputs,
                assistant_model=assistant_model,
                tokenizer=target_tokenizer,
                assistant_tokenizer=assistant_tokenizer
            )
            target_tokenizer.batch_decode(outputs, skip_special_tokens=True)

print("Assisted profiling data saved to:", logdir_assisted)
