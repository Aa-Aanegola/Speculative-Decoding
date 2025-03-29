# Things we need to do 

## Aakash 
### Figure out the compute situation
Running on a T4 is practically impossible - can't even load a 2b model into memory. Need to write code using dummy models and then for actual evaluation run on an A100 or something 
### Look into the huggingface transformers library - https://github.com/huggingface/transformers/blob/main/src/transformers/generation/candidate_generator.py
Gotta see if we can implement algo from below here
### Read - https://arxiv.org/pdf/2211.17192 see if we can lift the q(x | x_{<t>}) < p stuff
### Do we need to write our own benchmarking? 
sample 100 prompts that we can find average speedup over. There's literature for this, depending on model scope we can do some interesting stuff - do we want to push for a niche like code gen or sum? 
### Read - https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/
was cited in one of the huggingface things I was looking at, seems interesting. The more ideas we explore, the more we can experiment with 