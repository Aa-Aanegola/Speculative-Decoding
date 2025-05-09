# This file is to test if EAGLE is loaded correctly and can generate text 
# change the base_model_path to your own model path - sometimes it needs the snapshot path

import sys
sys.path.append('../EAGLE')
from eagle.model.ea_model import EaModel

import torch

base_model_path = "/insomnia001/depts/edu/COMSE6998/aa5506/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
EAGLE_model_path = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()
your_message="Hello"
conv = get_conversation_template("llama3")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])
print(output)

# from huggingface_hub import snapshot_download
# import os

# model_dir = snapshot_download("meta-llama/Llama-3.1-8B-Instruct")
# print(model_dir)
# print(os.listdir(model_dir))