import yaml
import sys
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from fsd import FSDAutoModelForCausalLM
import torch
sys.path.append('../../UmbreLLa')
# from umbrella.engine.auto_engine import AutoEngine
sys.path.append('../../')
# from Medusa.medusa.model.medusa_model import MedusaModelLlama, MedusaConfig

import json
logdir_base = 'logdir'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CHANGE ME
UNI = 'aa5506'

class Eagle2Wrapper:
    def __init__(self, model, draft_model, tokenizer, draft_tokenizer, config):
        self.model = model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.draft_tokenizer = draft_tokenizer
        self.config = config
        self.device = model.device

    def parameters(self):
        """Return all parameters from both target and draft models."""
        return list(self.model.parameters()) + list(self.draft_model.parameters())

    def children(self):
        """Return children modules from the target model."""
        return self.model.children()

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        max_new_tokens = kwargs.pop('max_new_tokens', 32)  # Remove from kwargs after getting value
        
        # Get Eagle2-specific parameters from config and remove from kwargs if present
        num_draft_tokens = self.config['generate_args'].get('num_draft_tokens', 5)
        acceptance_threshold = self.config['generate_args'].get('acceptance_threshold', 0.3)
        kwargs.pop('num_draft_tokens', None)
        kwargs.pop('acceptance_threshold', None)
        
        # Add do_sample=True if using sampling parameters
        if any(key in kwargs for key in ['temperature', 'top_k', 'top_p']):
            kwargs['do_sample'] = True
        
        current_ids = input_ids.clone()
        
        for _ in range(0, max_new_tokens, num_draft_tokens):
            # Generate draft tokens
            draft_outputs = self.draft_model.generate(
                current_ids,
                max_new_tokens=num_draft_tokens,
                pad_token_id=self.draft_tokenizer.pad_token_id,
                **kwargs
            )
            draft_tokens = draft_outputs[:, current_ids.shape[1]:]
            
            # Verify with target model
            logits = self.model(current_ids).logits[:, -1:]
            probs = torch.softmax(logits, dim=-1)
            
            # Accept tokens that meet threshold
            accepted_tokens = []
            for i, token in enumerate(draft_tokens[0]):
                if probs[0, -1, token.item()] >= acceptance_threshold:
                    accepted_tokens.append(token.item())
                else:
                    break
            
            if not accepted_tokens:
                # If no tokens accepted, generate one token with target model
                target_output = self.model.generate(
                    current_ids,
                    max_new_tokens=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
                new_token = target_output[:, -1:]
                current_ids = torch.cat([current_ids, new_token], dim=1)
            else:
                # Add accepted tokens
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([accepted_tokens], device=self.device)
                ], dim=1)
            
            # Check if we should stop
            if self.tokenizer.eos_token_id in current_ids[0]:
                break
        
        return current_ids

def load_model(config):
    if config["type"] == "eagle2":
        target_model = AutoModelForCausalLM.from_pretrained(
            config["target_model"], 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        target_tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        
        draft_model = AutoModelForCausalLM.from_pretrained(
            config["draft_model"], 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        draft_tokenizer = AutoTokenizer.from_pretrained(config["draft_model"])
        
        target_model.eval()
        draft_model.eval()
        
        wrapped_model = Eagle2Wrapper(
            target_model,
            draft_model,
            target_tokenizer,
            draft_tokenizer,
            config
        )
        
        return wrapped_model, target_tokenizer, {}
        
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
    
    elif config["type"] == "medusa-gemma":
        target_model = AutoModel.from_pretrained(config["target_model"])
        target_tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        target_model.eval()
        return target_model, target_tokenizer, {}
    
    else:
        raise ValueError(f"Unknown model type: {config['type']}")
    
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data