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

repo_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(repo_root))

print(f"Added repository root {repo_root} to sys.path")

def resolve_repo_path(config_path_str: str) -> Path:
    """Resolves a path string from the config file relative to the repository root or as an absolute path."""
    p = Path(config_path_str)
    if p.is_absolute():
        return p
    else:
        return repo_root / p

def resolve_current_dir_path(config_path_str: str) -> Path:
    """Resolves a path string from the config file relative to the current directory (routing/) or as an absolute path."""
    p = Path(config_path_str)
    if p.is_absolute():
        return p
    else:
        # The current directory is 'routing/'
        current_dir = Path(__file__).parent.resolve()
        return current_dir / p


print("Attempting to import necessary modules...")
try:
    from utils import load_model
    # from utils.fsd_utils import FSDAutoModelForCausalLM # Uncomment if needed 

    _FASTCHAT_AVAILABLE = False # Default to False
    try:
        from fastchat.model import get_conversation_template
        _FASTCHAT_AVAILABLE = True
        print("fastchat imported successfully.")
    except ImportError:
        print("Warning: fastchat not found. Eagle model conversation templating might not work as expected.")


    print("Necessary modules imported successfully.")
except ImportError as e:
    print(f"!Fatal Error: Could not import necessary modules. Make sure requirements are installed: {e}")
    traceback.print_exc()
    sys.exit(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
if device.type == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


models = {} # Dictionary to hold loaded models, tokenizers, and generation args {model_type: {model, tokenizer, generate_args}}
sentence_model = None # Sentence transformer model for clustering
kmeans = None # KMeans model for clustering
speedup_map = {} #  mapping cluster_id to best_model_type {cluster_id: model_type}
config = {} # Loaded routing configuration
conversation_history = [] # List to store chat history in interactive mode [{"role": str, "content": str}, ...]


print("Starting load_all_resources function definition.")
def load_all_resources():
    """Loads all models, tokenizers, clustering data, and speedup map based on routing config."""
    global models, sentence_model, kmeans, speedup_map, config

    print("Inside load_all_resources.")
    config_path = resolve_repo_path("configs/routing.yml")
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
        loaded_model_types = set() # Keep track of successfully loaded model types
        for cfg_path_relative_in_config in config['methods']:
            print(f"Attempting to load model config path listed in routing config: '{cfg_path_relative_in_config}'")

            if not isinstance(cfg_path_relative_in_config, str) or cfg_path_relative_in_config.strip() == "" or cfg_path_relative_in_config.strip() == "-":
                 print(f"Skipping invalid model config path string: '{cfg_path_relative_in_config}'")
                 continue

            cfg_filename = Path(cfg_path_relative_in_config).name
            cfg_path = repo_root / "configs" / cfg_filename 

            print(f"Resolved model config path to: {cfg_path.resolve()}")

            if not cfg_path.exists():
                 print(f"Warning: Model config file not found at {cfg_path.resolve()}. Skipping.")
                 continue

            try:
                with open(cfg_path, 'r') as file:
                    model_cfg = yaml.safe_load(file)
                model_type = model_cfg.get('type')
                if not model_type or not isinstance(model_type, str):
                    print(f"Warning: Model config file {cfg_path.resolve()} is missing or has an invalid 'type'. Skipping.")
                    continue

                if model_type in models:
                    print(f"Warning: Model type '{model_type}' already loaded. Skipping duplicate config: {cfg_path.resolve()}")
                    continue # Avoid loading the same model type multiple times

                print(f"Loading model type: {model_type}...")


                model, tokenizer, generate_args = load_model(model_cfg)
                print(f"Model {model_type} loaded successfully.")

                # Store loaded model components keyed by the model type
                models[model_type] = {
                    'model': model, 
                    'tokenizer': tokenizer,
                    'generate_args': {**generate_args, **model_cfg.get('generate_args', {})} # Merge default and config args
                }
                loaded_model_types.add(model_type)

            except Exception as e:
                print(f"!Error loading model from config '{cfg_path_relative_in_config}' (resolved to {cfg_path.resolve()}) (type: {model_type if 'model_type' in locals() else 'N/A'}): {e}")
                traceback.print_exc()


    print(f"Finished loading models. Successfully loaded types: {list(models.keys())}")

    # Load speedup data 
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
                # Process speedup data, mapping cluster IDs to the best loaded model type
                for cluster_k, method_v in speedup_data_raw.items():
                     try:
                         cluster_id = int(cluster_k)
                         # Filter methods in speedup data to only include those successfully loaded
                         loaded_methods_in_cluster = {m: score for m, score in method_v.items() if m in models}
                         if loaded_methods_in_cluster:
                             # Pick the best loaded method based on score (max score)
                             best_method_for_cluster = max(loaded_methods_in_cluster.items(), key=lambda x: x[1])[0]
                             speedup_map[cluster_id] = best_method_for_cluster
                         # If no loaded models listed for this cluster in speedup data, it will fall back to the overall default.
                     except ValueError:
                         print(f"Warning: Skipping non-integer cluster key in speedup data: '{cluster_k}'")
                     except Exception as e:
                          print(f"Warning: Error processing speedup data for cluster '{cluster_k}': {e}. Skipping.")

            except Exception as e:
                print(f"! Error loading or parsing speedup data from {speedup_path.resolve()}: {e}")
                traceback.print_exc()
                speedup_map = {}

    # Determine the default model type 
    # Priority: config default > first loaded model > 'naive_placeholder'
    default_model_type_from_config = config.get('interactive', {}).get('default')
    if default_model_type_from_config is not None and default_model_type_from_config in models:
         default_model_type = default_model_type_from_config
         print(f"Using config default model '{default_model_type}' as default.")
    elif models:
        # If config default failed or wasn't loaded, use the first successfully loaded model type
        default_model_type = list(models.keys())[0]
        print(f"Warning: Config default model '{default_model_type_from_config}' not loaded or specified. Using first loaded model '{default_model_type}' as default.")
    else:
        # If no models loaded at all, use a placeholder name (generation will fail)
        default_model_type = 'naive_placeholder' # Using a distinct placeholder name
        print(f"Warning: No models loaded successfully. Cannot set a valid default model type. Using placeholder '{default_model_type}'. Generation will likely fail.")

    speedup_map['default'] = default_model_type # Store the chosen default model type
    print(f"Default model type for routing: {speedup_map['default']}")

    # Load Clustering Data 
    kmeans_path_relative_in_config = config.get('interactive', {}).get('kmeans')
    if kmeans_path_relative_in_config is None:
        print("Warning: 'kmeans' key missing in interactive config. Clustering disabled.")
        sentence_model = None
        kmeans = None
    else:
        kmeans_filename = Path(kmeans_path_relative_in_config).name
        kmeans_path = repo_root / "clustering" / kmeans_filename 

        print(f"Loading clustering data from {kmeans_path.resolve()}")
        if not kmeans_path.exists():
            print(f"Warning: KMeans model file not found at {kmeans_path.resolve()}. Clustering disabled.")
            sentence_model = None
            kmeans = None
        else:
            try:
                # Load SentenceTransformer model first
                sentence_model_name = config.get('interactive', {}).get('sentence_model', 'all-MiniLM-L6-v2')
                print(f"Loading SentenceTransformer model '{sentence_model_name}'...")
                sentence_model = SentenceTransformer(sentence_model_name)
                print(f"SentenceTransformer model '{sentence_model_name}' loaded.")
                # Load KMeans model
                kmeans = joblib.load(kmeans_path)
                print("KMeans model loaded.")
            except Exception as e:
                print(f"! Error loading clustering models (KMeans from {kmeans_path.resolve()} or SentenceTransformer '{sentence_model_name}'): {e}")
                traceback.print_exc()
                sentence_model = None
                kmeans = None

    print("load_all_resources finished.")


@torch.no_grad()
def _generate(
    model_info: Dict,
    conversation_history: List[Dict],
    user_prompt: str,
    max_new_tokens: int,
) -> Dict:
    """
    Generates a response using the specified model and conversation history, similar to main.py's _generate.

    Args:
        model_info: Dict containing 'model', 'tokenizer', 'generate_args' (preloaded components).
        conversation_history: List of dicts {"role": str, "content": str}
        user_prompt: The current user's message.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Dict with 'output', 'total_time', 'tokens_per_second', 'total_generated_tokens_count'.
    """
    model = model_info['model']
    tokenizer = model_info['tokenizer']
    gen_args = model_info['generate_args'].copy()


    # Combine history and current prompt for templating.
    full_messages_for_templating = conversation_history + [{"role": "user", "content": user_prompt}]

    # Apply chat template
    # Use a reasonable number of past turns for context
    max_history_turns = config.get('interactive', {}).get('max_history_turns', 3)
    # Keep the user's current prompt and the last N turns of history (N user + N assistant messages)
    prompt_messages_for_templating = full_messages_for_templating[-(max_history_turns * 2 + 1):]

    final_prompt_string = ""
    try:
        # Attempt to apply chat template if the tokenizer supports it
        if hasattr(tokenizer, 'apply_chat_template'):
             final_prompt_string = tokenizer.apply_chat_template(
                prompt_messages_for_templating,
                tokenize=False,
                add_generation_prompt=True # Important for instruction-following models
            )
        elif model.__class__.__name__ == "EaModel" and _FASTCHAT_AVAILABLE:
             # Specific templating for Eagle model using fastchat if available
             conv = get_conversation_template("llama3") 
             for msg in prompt_messages_for_templating:
                  conv.append_message(str(msg['role']), str(msg['content']))
             if prompt_messages_for_templating and prompt_messages_for_templating[-1]['role'] != 'assistant':
                  conv.append_message(str(conv.roles[1]), None)
             final_prompt_string = conv.get_prompt()
        else:
            # Fallback for tokenizers without apply_chat_template or if fastchat not available for Eagle
            print(f"Warning: Tokenizer for {model_info.get('type', 'UnknownModel')} lacks apply_chat_template or fastchat not available for Eagle. Falling back to simple join.")
            final_prompt_string = ""
            for msg in prompt_messages_for_templating:
                final_prompt_string += f"{str(msg['role']).capitalize()}: {str(msg['content'])}\n"
            final_prompt_string += "Assistant:"


    except Exception as e:
        print(f"Warning: Could not apply chat template for {model_info.get('type', 'UnknownModel')} (Error: {e}). Falling back to simple join.")
        traceback.print_exc()
        final_prompt_string = ""
        for msg in prompt_messages_for_templating:
            final_prompt_string += f"{str(msg['role']).capitalize()}: {str(msg['content'])}\n"
        final_prompt_string += "Assistant:"


    # Tokenize the final prepared prompt string
    inputs = tokenizer(final_prompt_string, return_tensors="pt", truncation=True).to(model.device)

    gen_args['max_new_tokens'] = max_new_tokens

    if 'do_sample' not in gen_args:
         gen_args['do_sample'] = False # Default to greedy unless sampling is specified
    if 'temperature' in gen_args and gen_args['temperature'] <= 0:
         gen_args['do_sample'] = False # Disable sampling if temperature is non-positive
         for sampling_arg in ['top_k', 'top_p', 'typical_p', 'epsilon_cutoff', 'eta_cutoff', 'min_p']:
              if sampling_arg in gen_args:
                   gen_args.pop(sampling_arg)


    torch.cuda.synchronize()
    start = now()

    output = None
    try:
        if model.__class__.__name__ == "EaModel" and _FASTCHAT_AVAILABLE:
             output = model.eagenerate(inputs.input_ids, **gen_args)
        else:
             # This works for naive, spec-decoding, and FSD wrappers
             output = model.generate(
                **inputs,
                **gen_args
            )
    except Exception as e:
        print(f"Error during model generation: {e}")
        traceback.print_exc()
        return {
            'output': f"Error during generation: {e}",
            'total_time': 0.0,
            'tokens_per_second': 0.0,
            'total_generated_tokens_count': 0
        }


    torch.cuda.synchronize()
    end = now()

    # Decode the generated output
    # The output tensor contains the input_ids concatenated with the generated tokens
    input_ids_tensor = inputs["input_ids"]
    input_len = input_ids_tensor.shape[-1]

    output_tensor = output 

    if output_tensor.shape[-1] > input_len:
        generated_tokens_tensor = output_tensor[0, input_len:]
        decoded = tokenizer.decode(generated_tokens_tensor, skip_special_tokens=True)
        total_tokens_count = generated_tokens_tensor.shape[-1]
    else:
        decoded = ""
        total_tokens_count = 0

    cleaned_decoded = decoded.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|file_separator|>", "").strip()


    total_time = end - start
    tokens_per_second = total_tokens_count / total_time if total_time > 0 and total_tokens_count > 0 else 0.0

    return {
        'output': cleaned_decoded, # Only the new text generated by the model
        'total_time': total_time, # Time taken for this generation call
        'tokens_per_second': tokens_per_second, # Speed based on new tokens
        'total_generated_tokens_count': total_tokens_count # Count of new tokens
    }


print("Flask app setup starting.")
app = Flask(__name__)

# Load resources when the app starts
print("Attempting to load all resources...")
try:
    with app.app_context():
        load_all_resources()
    print("Resource loading phase complete.")
    # Check again if models were loaded successfully after trying
    if not models:
        print("!!!!!! No models were loaded successfully. The application will not be able to generate responses. !!!!!!")
except Exception as e:
    print(f"!!!!!! Fatal Error during initial resource loading: {e} !!!!!!")
    traceback.print_exc()
    print("Application cannot start due to fatal loading error.")
    sys.exit(1)


@app.route('/')
def index():
    """Serves the main chat interface page."""
    global conversation_history
    # Clear history when the index page is loaded/reloaded
    conversation_history = []
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handles incoming chat prompts from the frontend."""
    global conversation_history

    # Check if any models are loaded before processing a chat request
    if not models:
         err_msg = 'Error: No models were loaded successfully. Application cannot generate responses.'
         print(f"!!!!!! {err_msg} !!!!!!")
         return jsonify({'response': err_msg,
                         'metrics': {'error': 'No models loaded'}}), 500 # Return 500 Internal Server Error


    data = request.get_json()
    user_prompt = data.get('prompt') # Extract the user's message string.

    # Validate that a prompt was provided.
    if not user_prompt:
        return jsonify({'response': 'Error: No prompt provided.',
                         'metrics': {'error': 'No prompt provided'}}), 400


    # Routing Logic 
    # Determine the best model type based on clustering the user's prompt and looking up in the speedup data
    cluster_id_display = 'N/A'
    best_model_type = speedup_map.get('default', 'naive_placeholder')


    # Perform clustering if models are available and loaded
    if sentence_model is not None and kmeans is not None:
        try:
            # Embed the current user prompt using the SentenceTransformer model.
            prompt_embedding = sentence_model.encode(user_prompt)
            # Predict the cluster ID for this embedding using the KMeans model
            cluster_id_int = int(kmeans.predict([prompt_embedding])[0])
            cluster_id_display = str(cluster_id_int)
            print(f"User Prompt: '{user_prompt[:50]}...' (truncated), Cluster: {cluster_id_int}")
            # Look up the best model for this cluster, falling back to the overall default if not found
            best_model_type = speedup_map.get(cluster_id_int, speedup_map['default'])

        except Exception as e:
            print(f"Error during clustering: {e}")
            traceback.print_exc()
            cluster_id_display = 'clustering_failed'
            best_model_type = speedup_map['default'] # Ensure we use the default if clustering fails


    else:
        print("Clustering models not loaded. Using default model type for routing.")


    # Model Selection -
    if best_model_type not in models:
         print(f"Warning: Resolved model type '{best_model_type}' is not loaded (possibly due to a previous loading error).")
         # Fallback logic
         if models:
             best_model_type = list(models.keys())[0]
             print(f"Falling back to first loaded model type: '{best_model_type}'")
         else:
             print(f"No models loaded to fall back to. Using placeholder model type '{best_model_type}'. Generation will likely fail.")

    if best_model_type not in models:
         err_msg = f'Fatal: Selected model type "{best_model_type}" is not loaded. Cannot generate response.'
         print(f"!!!!!! {err_msg} !!!!!!")
         traceback.print_exc()
         return jsonify({'response': f'Error: {err_msg}',
                         'metrics': {'error': err_msg}}), 500 # 500 Internal Server Error status


    model_info = models[best_model_type]
    print(f"Using model: {best_model_type}, Cluster: {cluster_id_display}")


    #  Generating Response 
    try:
        max_new_tokens_config = config.get('interactive', {}).get('max_new_tokens', 512)

        # Call the unified _generate function
        response_data = _generate(
            model_info,
            conversation_history,
            user_prompt,
            max_new_tokens_config,
        )

        # Extract data from the response
        assistant_response_text = response_data.get('output', 'Error: No output generated.')
        total_time = response_data.get('total_time', 0.0)
        tokens_per_second = response_data.get('tokens_per_second', 0.0)
        generated_tokens_count = response_data.get('total_generated_tokens_count', 0)

        # Log generation details
        print(f"Generated {generated_tokens_count} tokens in {total_time:.3f}s ({tokens_per_second:.2f} tok/s) using {best_model_type}, Cluster: {cluster_id_display}")


        # Update and Truncate History 
        # Append the current user prompt and the assistant's response
        conversation_history.append({"role": "user", "content": user_prompt})
        conversation_history.append({"role": "assistant", "content": assistant_response_text})

        # Truncate history to prevent it from growing indefinitely
        MAX_OVERALL_HISTORY_TURNS = config.get('interactive', {}).get('max_overall_history_turns', 20)
        # Keep only the latest turns (each turn is two messages: user + assistant)
        if len(conversation_history) > MAX_OVERALL_HISTORY_TURNS * 2:
             conversation_history = conversation_history[-(MAX_OVERALL_HISTORY_TURNS * 2):]
             print(f"Truncated history to {MAX_OVERALL_HISTORY_TURNS} turns.")


        # Send Response back to Frontend 
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
        print(f"Error during generation using model '{best_model_type}': {e}")
        traceback.print_exc()
        return jsonify({'response': f'Error during generation: {e}',
                        'metrics': {'error': str(e), 'model_type_attempted': best_model_type}}), 500


if __name__ == '__main__':
    print("Flask app starting...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)