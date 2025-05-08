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