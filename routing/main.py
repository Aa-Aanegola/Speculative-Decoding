import argparse 
import yaml
import sys
sys.path.append('../benchmarks/')
from utils import *
from sentence_transformers import SentenceTransformer
import joblib
from time import perf_counter as now
import wandb
from datetime import datetime
import os

def log_to_wandb(entry_result, _id, cluster, model_type):
    wandb.log({
        "id": _id,
        "cluster": cluster,
        "num_tokens": len(entry_result["output"]),
        "total_time": entry_result["total_time"],
        "tokens_per_second": entry_result["tokens_per_second"],
        "model_type": model_type
    })

@torch.no_grad()
def _generate(model, tokenizer, generate_args, prompt: str, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    new_args = {**generate_args, "max_new_tokens": max_new_tokens}

    torch.cuda.synchronize()
    start = now()

    output = model.generate(
        **inputs,
        **new_args
    )

    torch.cuda.synchronize()
    end = now()

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    total_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
    total_time = end - start

    return {
        "output": decoded[len(prompt):],
        "total_time": total_time,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spec Decoding Router")
    parser.add_argument("--yaml", type=str, default="./configs/routing.yml", help="Path to the YAML config file")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "benchmark"], help="Mode to run the script in")
    
    args = parser.parse_args()
    
    with open(args.yaml, 'r') as file:
        config = yaml.safe_load(file)
    
    models = {}
    
    for cfg_path in config['methods']:
        with open(cfg_path, 'r') as file:
            model_cfg = yaml.safe_load(file)
        
        model, tokenizer, generate_args = load_model(model_cfg)
        models[model_cfg['type']] = {
            'model': model,
            'tokenizer': tokenizer,
            'generate_args': {**generate_args, **model_cfg['generate_args']}
        }
    
    speedup = json.load(open(config['speedup'], 'r'))
    
    speedup = {int(k): max(v.items(), key=lambda x: x[1])[0] for k, v in speedup.items()} 
    
    if args.mode == "interactive":
        sentence_model = SentenceTransformer(config['interactive']['sentence_model'])
        kmeans = joblib.load(config['interactive']['kmeans'])
        
        speedup['default'] = config['interactive']['default']
        
        while True:
            prompt = input("Enter your prompt: ")
            if prompt.lower() == "exit":
                break
            
            # Get the cluster for the prompt
            prompt_embedding = sentence_model.encode(prompt)
            cluster = int(kmeans.predict([prompt_embedding])[0])
            print(f"Prompt: {prompt}, Cluster: {cluster}")
                
            if cluster not in speedup:
                print(f"Cluster {cluster} not found in speedup data, using default.")
                cluster = 'default'
            else: 
                print(f"Cluster: {cluster}, using model type: {speedup[cluster]}")
            
            best_model_type = speedup[cluster]
            model_info = models[best_model_type]
            
            print(f"Using model type: {best_model_type}")
            
            # Generate response using the best model
            response = _generate(
                model_info['model'],
                model_info['tokenizer'],
                model_info['generate_args'],
                prompt,
                max_new_tokens=config['interactive']['max_new_tokens']
            )
            print(f"Response: {response['output']}")
            print(f"Total time: {response['total_time']:.3f}s, Tokens per second: {response['tokens_per_second']:.2f} tok/s")
    
    elif args.mode == "benchmark":  
        data = []
        with open(config['benchmark']['dataset'], 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        run = wandb.init(
            entity="speculative-decoding",
            project="full-profiles",
            config=config,
            name='routing (fsd vs. specd)'+ '_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )   
        
        results = []
        for entry in data:
            _id = entry["id"]
            prompt = entry["prompt"]
            result = _generate(
                models[speedup[entry['cluster']]]['model'],
                models[speedup[entry['cluster']]]['tokenizer'],
                models[speedup[entry['cluster']]]['generate_args'],
                prompt,
                max_new_tokens=config['benchmark']['max_new_tokens']
            )
            
            results.append({
                **result,
                **entry,
                "model_type": speedup[entry['cluster']]
            })
            
            print(f"[{_id} using {speedup[entry['cluster']]}] {len(result['output'])} tokens, {result['total_time']:.3f}s total, {result['tokens_per_second']:.2f} tok/s")
            log_to_wandb(result, _id, entry['cluster'], speedup[entry['cluster']])
        
        results_file = 'results_routing.json'
        with open(results_file, 'w') as f:
            json.dump(res, f, indent=4)
        wandb.finish()