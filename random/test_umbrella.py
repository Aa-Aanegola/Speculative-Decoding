from umbrella.engine.auto_engine import AutoEngine
import json

with open("umb_config_llama.json") as f:
    config = json.load(f)

DEVICE = "cuda:0"
engine = AutoEngine.from_config(device=DEVICE, **config)
engine.initialize()

GEN_LEN = 512
text1 = "Tell me what you know about Reinforcement Learning in 100 words."
text2 = "Tell me what you know about LSH in 100 words."

engine.prefill(text1)
engine.speculative_decoding(max_new_tokens=GEN_LEN)

engine.append(text2)
engine.speculative_decoding(max_new_tokens=GEN_LEN)