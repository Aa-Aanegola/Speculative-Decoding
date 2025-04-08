import yaml
import sys
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer
from fsd import FSDAutoModelForCausalLM
import torch

logdir_base = 'logdir'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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