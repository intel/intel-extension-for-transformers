
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex
import time

# load model
model_id = "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = model.to('cuda')
model = model.eval()

# to channels last
model = model.to(memory_format=torch.channels_last)
# to ipex
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

# input prompt
prompt = "Once upon a time,"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# start
elapsed = time.time()
with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    gen_tokens = model.generate(input_ids, do_sample=True,
                                temperature=0.9, max_length=32, num_beams=1)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
elapsed = time.time() - elapsed

print(gen_text)
print("inference Latency: %.3f ms." % (elapsed * 1000))

