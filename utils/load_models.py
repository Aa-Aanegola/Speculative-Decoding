import yaml
import sys
import os
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
sys.path.append('../../UmbreLLa')
sys.path.append('../../Medusa')
sys.path.append('../../EAGLE')

import json
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(config):
    if config["type"] == "eagle2":
        from eagle.model.ea_model import EaModel
        model = EaModel.from_pretrained(
            base_model_path=config['target_model'],
            ea_model_path=config['eagle_path'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=-1
        )
        
        
        return model, model.tokenizer, {}
        
    elif config["type"] == "spec-decoding":
        target_model = AutoModelForCausalLM.from_pretrained(config["target_model"], torch_dtype=torch.bfloat16, device_map="auto")
        target_tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        draft_model = AutoModelForCausalLM.from_pretrained(config["draft_model"], torch_dtype=torch.bfloat16, device_map='cuda')
        draft_tokenizer = AutoTokenizer.from_pretrained(config["draft_model"])
        target_model.eval()
        draft_model.eval()
    
        return target_model, target_tokenizer, {
            'assistant_model': draft_model,
            'pad_token_id': target_tokenizer.eos_token_id
        }
    elif config["type"] == "fsd":
        from utils import FSDAutoModelForCausalLM
        target_model = FSDAutoModelForCausalLM.from_pretrained(config["target_model"], torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        draft_model = FSDAutoModelForCausalLM.from_pretrained(config["draft_model"], torch_dtype=torch.bfloat16, device_map='cuda')
        target_model.eval()
        draft_model.eval()
        
        return target_model, tokenizer, {
            'assistant_model': draft_model,
            'pad_token_id': tokenizer.eos_token_id
        }
    elif config["type"] == "naive":
        target_model = AutoModelForCausalLM.from_pretrained(config["target_model"], torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        target_model.eval()
        
        return target_model, tokenizer, {
            'pad_token_id': tokenizer.eos_token_id
        }
    elif config['type'] == 'medusa-hf':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Nexesenex/Llama_3.1_8b_Medusa_v1.01")
        target_model.eval()
        
        return model, tokenizer, {
            'pad_token_id': tokenizer.eos_token_id
        }
        
    # These are commented out because they are not used in the current code - uncomment for more techniques to benchmark
    
    # elif config["type"] == "umbrella":
    #     with open(config["json"]) as f:
    #         json_data = json.load(f)
    #     target_model = AutoEngine.from_config(device=device, **json_data)
    # elif config["type"] == "umbrella":
    #     with open(config["json"]) as f:
    #         json_data = json.load(f)
    #     target_model = AutoEngine.from_config(device=device, **json_data)
        
    #     return target_model, target_model.tokenizer, {}

    # elif config["type"] == "medusa":
    #     target_model = MedusaModelLlama.from_pretrained(config["target_model"], torch_dtype=torch.bfloat16, device_map="auto")
    #     target_tokenizer = target_model.get_tokenizer()
    #     target_model.eval()
        
    #     # Create a wrapper that returns a tensor instead of a generator
    #     original_generate = target_model.medusa_generate
    #     def wrapped_generate(**kwargs):
    #         outputs = list(original_generate(**kwargs))
    #         final_text = outputs[-1]["text"]
    #         return target_tokenizer(final_text, return_tensors='pt').input_ids.to(device)
            
    #     target_model.generate = wrapped_generate
        
    #     return target_model, target_tokenizer, {}
    
#     elif config["type"] == "medusa-gemma":
#         target_model = AutoModel.from_pretrained(config["target_model"])
#         target_tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
#         target_model.eval()
#         return target_model, target_tokenizer, {}
#         # Load the model directly - the MedusaModel class will handle config creation
#         model = MedusaModel.from_pretrained(
#             config["target_model"]["medusa"],
#             torch_dtype=torch.float16,
#             device_map="auto",
#             trust_remote_code=True
#         )
        
#         # Use base model tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(
#             config["target_model"]["base"],
#             trust_remote_code=True
#         )
#         model.eval()
        
#         # Get max tokens from config
#         max_steps = config["generate_args"].get("max_new_tokens", config["generate_args"].get("max_length", 512))
        
#         # Use the predefined medusa_choices for vicuna-7b
#         generation_kwargs = {
#             "temperature": config["generate_args"].get("temperature", 0.7),
#             "max_steps": max_steps,
#             "posterior_threshold": config["generate_args"].get("posterior_threshold", 0.09),
#             "posterior_alpha": config["generate_args"].get("posterior_alpha", 0.3),
#             "medusa_choices": vicuna_7b_stage2,  # Use the predefined choices
#             "top_p": config["generate_args"].get("top_p", 0.8),
#             "sampling": config["generate_args"].get("sampling", "typical"),
#             "fast": config["generate_args"].get("fast", True)
#         }
        
#         return model, tokenizer, generation_kwargs
#   
    else:
        raise ValueError(f"Unknown model type: {config['type']}")