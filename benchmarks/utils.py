import yaml
import sys
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer
from fsd import FSDAutoModelForCausalLM
import torch
sys.path.append('../../UmbreLLa')
# from umbrella.engine.auto_engine import AutoEngine
sys.path.append('../../')
from Medusa.medusa.model.medusa_model import MedusaModelLlama, MedusaConfig

import json
logdir_base = 'logdir'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CHANGE ME
UNI = 'aa5506'

def load_model(config):
    if config["type"] == "spec-decoding":
        target_model = AutoModelForCausalLM.from_pretrained(config["target_model"], torch_dtype=torch.bfloat16, device_map="auto")
        target_tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        draft_model = AutoModelForCausalLM.from_pretrained(config["draft_model"], torch_dtype=torch.bfloat16, device_map='cuda')
        draft_tokenizer = AutoTokenizer.from_pretrained(config["draft_model"])
        target_model.eval()
        draft_model.eval()
        
    
        return target_model, target_tokenizer, {
            'assistant_model': draft_model
        }
    elif config["type"] == "fsd":
        target_model = FSDAutoModelForCausalLM.from_pretrained(config["target_model"], torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        draft_model = FSDAutoModelForCausalLM.from_pretrained(config["draft_model"], torch_dtype=torch.bfloat16, device_map='cuda')
        target_model.eval()
        draft_model.eval()
        
        
        return target_model, tokenizer, {
            'assistant_model': draft_model
        }
    elif config["type"] == "naive":
        target_model = AutoModelForCausalLM.from_pretrained(config["target_model"], torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        target_model.eval()
        
        return target_model, tokenizer, {}
    
    # elif config["type"] == "umbrella":
    #     with open(config["json"]) as f:
    #         json_data = json.load(f)
    #     target_model = AutoEngine.from_config(device=device, **json_data)
    # elif config["type"] == "umbrella":
    #     with open(config["json"]) as f:
    #         json_data = json.load(f)
    #     target_model = AutoEngine.from_config(device=device, **json_data)
        
    #     return target_model, target_model.tokenizer, {}

    elif config["type"] == "medusa":
        target_model = MedusaModelLlama.from_pretrained(config["target_model"], torch_dtype=torch.bfloat16, device_map="auto")
        target_tokenizer = target_model.get_tokenizer()
        target_model.eval()
        
        # Create a wrapper that returns a tensor instead of a generator
        original_generate = target_model.medusa_generate
        def wrapped_generate(**kwargs):
            outputs = list(original_generate(**kwargs))
            final_text = outputs[-1]["text"]
            return target_tokenizer(final_text, return_tensors='pt').input_ids.to(device)
            
        target_model.generate = wrapped_generate
        
        return target_model, target_tokenizer, {}
    
    else:
        raise ValueError(f"Unknown model type: {config['type']}")
    
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data