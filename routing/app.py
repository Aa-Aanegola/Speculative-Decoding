import argparse
import yaml
import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, g
from sentence_transformers import SentenceTransformer
import joblib
from time import perf_counter as now
import torch
import json
from typing import List, Dict
import traceback 

def resolve_repo_path(config_path_str: str) -> Path:
    """Resolves a path string from the config file relative to the repository root or as an absolute path."""
    p = Path(config_path_str)
    if p.is_absolute():
        return p
    else:
        #  this script (app.py) should be located in root/routing/
        repo_root = Path(__file__).parent.parent
        return repo_root / p

def resolve_current_dir_path(config_path_str: str) -> Path:
    """Resolves a path string from the config file relative to the current directory or as an absolute path."""
    p = Path(config_path_str)
    if p.is_absolute():
        return p
    else:
        #  this script (app.py) should be located in root/routing/
        current_dir = Path(__file__).parent
        return current_dir / p

BENCHMARKS_PATH = Path(__file__).parent.parent / "benchmarks"
sys.path.append(str(BENCHMARKS_PATH))
print(f"Adding {BENCHMARKS_PATH.resolve()} to sys.path")
print("Attempting to import load_model...")
try:
    from utils import load_model 
    print("load_model imported successfully.")
except ImportError as e:
    print(f"!Error importing load_model from {BENCHMARKS_PATH.resolve() / 'utils.py'}: {e}")
    sys.exit(1) #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
if device.type == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


models = {}
sentence_model = None
kmeans = None 
speedup_map = {} # cluster_id -> best_model_type
config = {}
conversation_history = [] # history in memory


print("Starting load_all_resources function definition.")
def load_all_resources():
    """Loads all models, tokenizers, clustering data, and speedup map."""
    global models, sentence_model, kmeans, speedup_map, config

    print("Inside load_all_resources.")
    #  routing.yml should be in root/routing/configs/
    config_path = Path(__file__).parent / "configs" / "routing.yml"
    print(f"Loading routing config from {config_path.resolve()}")
    if not config_path.exists():
        raise FileNotFoundError(f"Routing config not found at {config_path.resolve()}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Routing config loaded.")

    except Exception as e:
        print(f"!Error loading or parsing routing config from {config_path.resolve()}: {e}")
        traceback.print_exc()
        sys.exit(1) 


    print("Loading models...")
    if 'methods' not in config or not isinstance(config['methods'], list) or not config['methods']:
        print("Warning: 'methods' key missing, not a list, or empty in routing.yml. No models configured to load.")
        models = {} 
    else:
        loaded_model_types = set() # track of successfully loaded model types
        for cfg_path_relative in config['methods']:
            print(f"Attempting to load model config path: '{cfg_path_relative}'")
            if not isinstance(cfg_path_relative, str) or cfg_path_relative.strip() == "" or cfg_path_relative.strip() == "-":
                 print(f"Skipping invalid model config path string: '{cfg_path_relative}'")
                 continue 

            try:
                cfg_path = resolve_repo_path(cfg_path_relative)
                print(f"Resolved model config path to: {cfg_path.resolve()}")

                if not cfg_path.exists():
                     print(f"Warning: Model config file not found at {cfg_path.resolve()}. Skipping.")
                     continue

                with open(cfg_path, 'r') as file:
                    model_cfg = yaml.safe_load(file)
                model_type = model_cfg.get('type')
                if not model_type or not isinstance(model_type, str):
                    print(f"Warning: Model config file {cfg_path.resolve()} is missing or has an invalid 'type'. Skipping.")
                    continue

                if model_type in models:
                    print(f"Warning: Model type '{model_type}' already loaded from a previous config. Skipping duplicate config: {cfg_path.resolve()}")
                    continue # Avoid loading the same model type multiple times
                    
                print(f"Loading model type: {model_type} from {model_cfg.get('target_model', model_cfg.get('model', 'N/A'))}...")

                model, tokenizer, generate_args = load_model(model_cfg)
                print(f"Model {model_type} loaded successfully.")

                # Store loaded model components keyed by the model type
                models[model_type] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'generate_args': {**generate_args, **model_cfg.get('generate_args', {})}
                }
                loaded_model_types.add(model_type) 

            except Exception as e:
                print(f"!Error loading model from config '{cfg_path_relative}' (type: {model_type if 'model_type' in locals() else 'N/A'}): {e}")
                traceback.print_exc()


    print(f"Finished loading models. Successfully loaded types: {list(models.keys())}")

    # loading speedup data
    speedup_path_relative = config.get('speedup') 
    if speedup_path_relative is None:
         print("Warning: 'speedup' key missing in routing config. Routing will use the default model for all clusters.")
         speedup_map = {}
    else:
        speedup_path = resolve_current_dir_path(speedup_path_relative)
        print(f"Loading speedup data from {speedup_path.resolve()}")
        if not speedup_path.exists():
             print(f"Warning: Speedup data file not found at {speedup_path.resolve()}. Routing will use the default model for all clusters.")
             speedup_map = {} 
        else:
            try:
                with open(speedup_path, 'r') as f:
                    speedup_data_raw = json.load(f)
                speedup_map = {} 
                for cluster_k, method_v in speedup_data_raw.items():
                     try:
                         cluster_id = int(cluster_k)
                         loaded_methods_in_cluster = {m: score for m, score in method_v.items() if m in models}
                         if loaded_methods_in_cluster:
                             # Pick the best loaded method based on score (max score)
                             best_method_for_cluster = max(loaded_methods_in_cluster.items(), key=lambda x: x[1])[0]
                             speedup_map[cluster_id] = best_method_for_cluster
                             # print(f"Cluster {cluster_id} mapped to {speedup_map[cluster_id]} (best loaded method from speedup data).")
                     except ValueError:
                         print(f"Warning: Skipping non-integer cluster key in speedup data: '{cluster_k}'")
                     except Exception as e:
                          print(f"Warning: Error processing speedup data for cluster '{cluster_k}': {e}. Skipping.")


            except Exception as e:
                print(f"! Error loading or parsing speedup data from {speedup_path.resolve()}: {e}")
                traceback.print_exc()
                speedup_map = {} 


    default_model_type = config.get('interactive', {}).get('default')
    if default_model_type is None or not isinstance(default_model_type, str) or default_model_type not in models:
        if models:
            default_model_type = list(models.keys())[0]
            print(f"Warning: Config default model '{config.get('interactive', {}).get('default')}' not loaded or specified. Using first loaded model '{default_model_type}' as default.")
        else:
            default_model_type = 'naive'
            print(f"Warning: No models loaded, cannot set a valid default model type. Using '{default_model_type}'.")


    speedup_map['default'] = default_model_type
    print(f"Default model type for routing: {speedup_map['default']}")

    #  Load Clustering Data 
    kmeans_path_relative = config.get('interactive', {}).get('kmeans')
    if kmeans_path_relative is None:
        print("Warning: 'kmeans' key missing in interactive config. Clustering disabled.")
        sentence_model = None
        kmeans = None
    else:
        kmeans_path = resolve_repo_path(kmeans_path_relative)
        print(f"Loading clustering data from {kmeans_path.resolve()}")
        if not kmeans_path.exists():
            print(f"Warning: KMeans model file not found at {kmeans_path.resolve()}. Clustering disabled.")
            sentence_model = None
            kmeans = None
        else:
            try:
                sentence_model_name = config.get('interactive', {}).get('sentence_model', 'all-MiniLM-L6-v2')
                print(f"Loading SentenceTransformer model '{sentence_model_name}'...")
                sentence_model = SentenceTransformer(sentence_model_name)
                print(f"SentenceTransformer model '{sentence_model_name}' loaded.")
                kmeans = joblib.load(kmeans_path)
                print("KMeans model loaded.")
            except Exception as e:
                print(f"! Error loading clustering models from {kmeans_path.resolve()} or SentenceTransformer '{sentence_model_name}': {e}")
                traceback.print_exc()
                sentence_model = None
                kmeans = None


    print("load_all_resources finished.")


@torch.no_grad()
def generate_response(
    model_info: Dict, 
    conversation_history: List[Dict], 
    user_prompt: str, 
    max_new_tokens: int, 
    model = None,
    tokenizer = None,
    generate_args = None
) -> Dict:
    """
    Generates a response using the specified model and conversation history.

    Args:
        model_info: Dict containing 'model', 'tokenizer', 'generate_args'
        conversation_history: List of dicts {"role": str, "content": str}
        user_prompt: The current user's message.
        max_new_tokens: Maximum tokens to generate.
        model, tokenizer, generate_args: Directly passed model components.

    Returns:
        Dict with 'output', 'total_time', 'tokens_per_second', 'total_generated_tokens_count'.
    """
    model = model
    tokenizer = tokenizer
    generate_args = generate_args


    # Combine history and current prompt for templating.
    full_messages_for_templating = conversation_history + [{"role": "user", "content": user_prompt}]


    # Apply chat template
    max_history_turns = config.get('interactive', {}).get('max_history_turns', 3)
    prompt_messages_for_templating = full_messages_for_templating[-(max_history_turns * 2 + 1):]

    final_prompt_string = ""
    try:
        final_prompt_string = tokenizer.apply_chat_template(
            prompt_messages_for_templating,
            tokenize=False,
            add_generation_prompt=True
        )

    except Exception as e:
        print(f"Warning: Could not apply chat template for {model_info.get('type', 'UnknownModel')} (Error: {e}). Falling back to simple join.")
        traceback.print_exc()
        final_prompt_string = ""
        for msg in prompt_messages_for_templating:
            final_prompt_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
        final_prompt_string += "Assistant:"


    inputs = tokenizer(final_prompt_string, return_tensors="pt").to(model.device)

    gen_args = {**generate_args}
    gen_args['max_new_tokens'] = max_new_tokens


    torch.cuda.synchronize()
    start = now()

    output = model.generate(
        **inputs, 
        **gen_args
    )

    torch.cuda.synchronize()
    end = now()

    input_len = inputs["input_ids"].shape[-1]

    # Check if any new tokens were generated.
    if output.shape[-1] > input_len:
        generated_tokens_tensor = output[0, input_len:]
        decoded = tokenizer.decode(generated_tokens_tensor, skip_special_tokens=True)
        total_tokens_count = generated_tokens_tensor.shape[-1]
    else:
        decoded = ""
        total_tokens_count = 0

    cleaned_decoded = decoded.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|file_separator|>", "").strip()

    total_time = end - start
    tokens_per_second = total_tokens_count / total_time if total_time > 0 and total_tokens_count > 0 else 0.0


    return {
        'output': cleaned_decoded, # Only the new text generated by the model.
        'total_time': total_time, # Time taken for this generation call.
        'tokens_per_second': tokens_per_second, # Speed based on new tokens.
        'total_generated_tokens_count': total_tokens_count # Count of new tokens.
    }


print("Flask app setup starting.")
app = Flask(__name__)

# Load resources when the app starts
print("Attempting to load all resources...")
try:
    with app.app_context():
        load_all_resources()
    print("Resource loading phase complete.")
    if not models:
        print("!!!!!! No models were loaded successfully. The application will not be able to generate responses. !!!!!!")
        sys.exit(1)
except Exception as e:
    print(f"!!!!!! Fatal Error during initial resource loading: {e} !!!!!!")
    traceback.print_exc() 
    print("Application cannot start due to fatal loading error.")
    sys.exit(1) 


@app.route('/')
def index():
    """Serves the main chat interface page."""
    global conversation_history
    conversation_history = []
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handles incoming chat prompts from the frontend."""
    global conversation_history 

    # Check if any models are loaded before processing a chat request
    if not models:
         return jsonify({'response': 'Error: No models are loaded. Application cannot generate responses.',
                         'metrics': {'error': 'No models loaded'}}), 500 # Return 500 Internal Server Error


    data = request.get_json()
    user_prompt = data.get('prompt') # Extract the user's message string.

    # validating that a prompt was provided.
    if not user_prompt:
        return jsonify({'response': 'Error: No prompt provided.',
                         'metrics': {'error': 'No prompt provided'}}), 400


    # Routing Logic
    # Determine the best model type based on clustering the user's prompt and looking up in the speedup data
    cluster_id_display = 'N/A' 
    best_model_type = speedup_map.get('default')

    # Perform clustering 
    if sentence_model is not None and kmeans is not None:
        try:
            # Embed the current user prompt using the SentenceTransformer model.
            prompt_embedding = sentence_model.encode(user_prompt)
            # Predict the cluster ID for this embedding using the KMeans model
            cluster_id_int = int(kmeans.predict([prompt_embedding])[0])
            cluster_id_display = str(cluster_id_int)
            print(f"User Prompt: '{user_prompt[:50]}...' (truncated), Cluster: {cluster_id_int}")
            best_model_type = speedup_map.get(cluster_id_int, speedup_map['default'])

        except Exception as e:
            print(f"Error during clustering: {e}")
            traceback.print_exc() 
            cluster_id_display = 'clustering_failed' 

    else:
        print("Clustering models not loaded. Using default model type for routing.")


    #  Model Selection 
    if best_model_type not in models:
         print(f"Warning: Resolved model type '{best_model_type}' is not loaded. Falling back to default.")
         traceback.print_exc()
         best_model_type = speedup_map['default'] 

    #  ensuring the selected best_model_type is indeed in the loaded models dictionary
    if best_model_type not in models:
         err_msg = f'Fatal: Resolved model type "{best_model_type}" is not loaded. Cannot generate response.'
         print(f"!!!!!! {err_msg} !!!!!!")
         traceback.print_exc()
         return jsonify({'response': f'Error: {err_msg}', 
                         'metrics': {'error': err_msg}}), 500 # 500 Internal Server Error status


    model_info = models[best_model_type]

    # Generating Response
    try:
        max_new_tokens_config = config.get('interactive', {}).get('max_new_tokens', 512)

        response_data = generate_response(
            model_info, 
            conversation_history, 
            user_prompt, 
            max_new_tokens_config, 
            model=model_info['model'],
            tokenizer=model_info['tokenizer'],
            generate_args=model_info['generate_args']
        )

        # Extract data from the response returned by generate_response
        assistant_response_text = response_data['output']
        total_time = response_data['total_time']
        tokens_per_second = response_data['tokens_per_second']
        generated_tokens_count = response_data['total_generated_tokens_count']

        print(f"Generated {generated_tokens_count} tokens in {total_time:.3f}s ({tokens_per_second:.2f} tok/s) using {best_model_type}, Cluster: {cluster_id_display}")


        #  Update History
        # Append the current user prompt and the assistant's response to the global conversation history.
        conversation_history.append({"role": "user", "content": user_prompt})
        conversation_history.append({"role": "assistant", "content": assistant_response_text})

        # History Truncation 
        MAX_OVERALL_HISTORY_TURNS = config.get('interactive', {}).get('max_overall_history_turns', 20) # Configurable
        # If the total number of messages in the history exceeds the limit (each turn is two entries), truncate it.
        if len(conversation_history) > MAX_OVERALL_HISTORY_TURNS * 2:
            # Keeping only the latest turns
            conversation_history = conversation_history[-(MAX_OVERALL_HISTORY_TURNS * 2):]
            print(f"Truncated history to {MAX_OVERALL_HISTORY_TURNS} turns.")

        #  Response goes back to Frontend 
        return jsonify({
            'response': assistant_response_text,
            'metrics': {
                'total_time': total_time,
                'tokens_per_second': tokens_per_second,
                'model_type': best_model_type,
                'cluster': cluster_id_display, 
                'generated_tokens': generated_tokens_count
            }
        })

    except Exception as e:
        # catching any errors that occur during the generation process.
        print(f"Error during generation: {e}")
        traceback.print_exc() # Print the full traceback for debugging.
        #  error message to the frontend.
        return jsonify({'response': f'Error during generation: {e}',
                        'metrics': {'error': str(e)}}), 500 #  500 Internal Server Error status


if __name__ == '__main__':
    print("Flask app starting...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)