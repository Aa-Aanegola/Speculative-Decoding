# Testing the FSD (Fast Speculative Decoding) implementation with a small and large model
# Some changes had to be made to the fsd_utils file for this to work correctly which is why 
# we supply a local version.

import sys
sys.path.append('../')
from transformers import AutoTokenizer
from utils import FSDAutoModelForCausalLM
import torch

small_model_id = 'google/gemma-2-2b-it'
large_model_id = 'google/gemma-2-9b-it'

device = torch.device('cuda')

small_tokenizer = AutoTokenizer.from_pretrained(small_model_id)
small_model = FSDAutoModelForCausalLM.from_pretrained(small_model_id, torch_dtype=torch.bfloat16).to(device)
large_model = FSDAutoModelForCausalLM.from_pretrained(large_model_id, torch_dtype=torch.bfloat16, device_map='auto')

input_text = "Write me an essay about the massive risks of climate change."
input_ids = small_tokenizer(input_text, return_tensors='pt').to(device)

output = large_model.generate(**input_ids, assistant_model=small_model, fsd_div_threshold=0.4, fsd_div_type='js_div', max_new_tokens=250)

print(f"output: {output}")
print(f"output: {small_tokenizer.decode(output[0])}")