import yaml
import sys
import os
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from fsd import FSDAutoModelForCausalLM
import torch
sys.path.append('../../UmbreLLa')
sys.path.append('../../Medusa')
sys.path.append('../../EAGLE')
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template

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
        
        # Set up proper padding tokens
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token
        
        if not hasattr(self.model.config, 'pad_token_id') or self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.draft_model.config.pad_token_id = self.draft_tokenizer.pad_token_id

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
        current_mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        
        for _ in range(0, max_new_tokens, num_draft_tokens):
            # Generate draft tokens
            draft_outputs = self.draft_model.generate(
                current_ids,
                attention_mask=current_mask,
                max_new_tokens=num_draft_tokens,
                pad_token_id=self.draft_tokenizer.pad_token_id,
                **kwargs
            )
            draft_tokens = draft_outputs[:, current_ids.shape[1]:]
            
            # Verify with target model
            model_outputs = self.model(
                current_ids,
                attention_mask=current_mask,
                return_dict=True
            )
            logits = model_outputs.logits[:, -1:]
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
                    attention_mask=current_mask,
                    max_new_tokens=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
                new_token = target_output[:, -1:]
                current_ids = torch.cat([current_ids, new_token], dim=1)
                current_mask = torch.cat([current_mask, torch.ones_like(new_token)], dim=1)
            else:
                # Add accepted tokens
                new_tokens = torch.tensor([accepted_tokens], device=self.device)
                current_ids = torch.cat([current_ids, new_tokens], dim=1)
                current_mask = torch.cat([current_mask, torch.ones_like(new_tokens)], dim=1)
            
            # Check if we should stop
            if self.tokenizer.eos_token_id in current_ids[0]:
                break
        
        return current_ids

def load_model(config):
    if config["type"] == "eagle2":
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
    
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data