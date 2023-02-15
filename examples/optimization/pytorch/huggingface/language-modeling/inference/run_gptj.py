
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex
import time
import sys
import argparse


# args
parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
parser.add_argument('--precision', default='bf16', type=str, help="fp32 or bf16")
parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")
parser.add_argument('--greedy', action='store_true')
args = parser.parse_args()
print(args)

amp_enabled = True if args.precision != "fp32" else False
amp_dtype = torch.bfloat16 if args.precision != "fp32" else torch.float32

if args.greedy:
    generate_kwargs = dict(do_sample=False, temperature=0.9)
else:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# load model
model_id = "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval()

# to channels last
model = model.to(memory_format=torch.channels_last)
# to ipex
model = ipex.optimize(model, dtype=amp_dtype, inplace=True)

# input prompt
# prompt = "Once upon a time,"
# 32 tokens input
prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
         " She wanted to go to places and meet new people, and have fun."

# start
total_time = 0.0
num_iter = 10
num_warmup = 3
with torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        toc = time.time()
        print(gen_text, flush=True)
        if i >= num_warmup:
            total_time += (toc - tic)

print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))

