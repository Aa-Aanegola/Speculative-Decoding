import os
import json
import time
import yaml
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
import sys

# Add Medusa to Python path
sys.path.append("MEDUSA_PATH")
from Medusa.medusa.model.medusa_model import MedusaModelLlama

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_questions(questions_file):
    with open(questions_file, 'r') as f:
        return [json.loads(line) for line in f]

def main():
    # Load config
    config_path = "/insomnia001/home/nb3227/Speculative-Decoding/benchmarks/configs/medusa.yml"
    config = load_config(config_path)
    
    # Setup wandb
    wandb_dir = "/insomnia001/home/nb3227/scratch/nb3227/wandb"
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    
    wandb.init(
        project="speculative-decoding",
        config=config,
        name="medusa-eval",
        dir=wandb_dir
    )

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = MedusaModelLlama.from_pretrained(config["target_model"]).to(device)
    tokenizer = model.get_tokenizer()

    # Load questions
    questions_file = "/insomnia001/home/nb3227/Speculative-Decoding/benchmarks/data/question.jsonl"
    print(f"Loading questions from {questions_file}")
    questions = load_questions(questions_file)
    print(f"Processing {len(questions)} questions...")

    # Process questions
    results = []
    total_tokens = 0
    total_time = 0

    for question in tqdm(questions):
        question_id = question['question_id']
        turns = question['turns']
        first_turn = turns[0]  # We'll just use the first turn for now
        
        # Prepare input
        input_text = first_turn
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Generate
        start_time = time.time()
        output = ""
        try:
            for step in model.medusa_generate(
                input_ids,
                **config["generate_args"]
            ):
                output += step["text"]
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            continue
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Count output tokens
        output_tokens = len(tokenizer.encode(output))
        total_tokens += output_tokens
        total_time += generation_time
        
        # Log to wandb
        wandb.log({
            "question_id": question_id,
            "generation_time": generation_time,
            "output_tokens": output_tokens,
            "tokens_per_second": output_tokens / generation_time if generation_time > 0 else 0
        })
        
        # Store result
        result = {
            "question_id": question_id,
            "input": input_text,
            "output": output,
            "generation_time": generation_time,
            "output_tokens": output_tokens,
            "tokens_per_second": output_tokens / generation_time if generation_time > 0 else 0
        }
        results.append(result)

    # Calculate and log final metrics
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    avg_time_per_question = total_time / len(questions)

    wandb.log({
        "total_questions": len(questions),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_time_per_question": avg_time_per_question
    })

    print("\nEvaluation Results:")
    print(f"Total questions processed: {len(questions)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average tokens per second: {avg_tokens_per_second:.2f}")
    print(f"Average time per question: {avg_time_per_question:.2f} seconds")

    # Save results
    output_dir = "/insomnia001/home/nb3227/scratch/nb3227/medusa_results"
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "medusa_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "metrics": {
                "total_questions": len(questions),
                "total_tokens": total_tokens,
                "total_time": total_time,
                "avg_tokens_per_second": avg_tokens_per_second,
                "avg_time_per_question": avg_time_per_question
            }
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")
    wandb.finish()

if __name__ == "__main__":
    main()
