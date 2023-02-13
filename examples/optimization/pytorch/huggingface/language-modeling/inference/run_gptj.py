
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex
import time
import sys


# set args
if len(sys.argv) > 1 and sys.argv[1] not in ['fp32', 'float32']:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = None

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
prompt = "Once upon a time,"

# start
elapsed = time.time()
with torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, do_sample=True,
                                temperature=0.9, max_length=32, num_beams=1)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
elapsed = time.time() - elapsed

print(gen_text)
print("Inference latency: %.3f ms." % (elapsed * 1000))

