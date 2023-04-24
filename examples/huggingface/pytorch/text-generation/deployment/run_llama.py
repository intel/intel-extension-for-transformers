import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import time
import sys
import types
import pathlib
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
from accelerate import init_empty_weights
import generation_utils as itrex_generation_utils

# args
parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
parser.add_argument("--ir_path",
        type=str,
        help="path to bfloat16 or int8 IR files",
        default="bfloat16",
    )
parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")
parser.add_argument('--input-tokens', default='32', type=str)
parser.add_argument('--prompt', default=None, type=str)
parser.add_argument('--batch-size', default=1, type=int)
args = parser.parse_args()
print(args)

generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
generate_kwargs["past_kv_nums"] = 32
generate_kwargs["llama"] = True 
# load model
model_id = "decapoda-research/llama-13b-hf"
config = AutoConfig.from_pretrained(model_id)
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
setattr(model, "generate",  types.MethodType(itrex_generation_utils.GenerationMixin.generate, model))
setattr(model, "beam_search", types.MethodType(itrex_generation_utils.GenerationMixin.beam_search, model))
setattr(model, "_update_model_kwargs_for_generation",  types.MethodType(itrex_generation_utils.GenerationMixin._update_model_kwargs_for_generation, model))
setattr(model, "_get_stopping_criteria", types.MethodType(itrex_generation_utils.GenerationMixin._get_stopping_criteria, model))
setattr(model, "_extract_past_from_model_output", types.MethodType(itrex_generation_utils.GenerationMixin._extract_past_from_model_output, model))
model.eval()

# input prompt
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + '/llamaprompt.json') as f:
    prompt_pool = json.load(f)
if args.prompt is not None:
    prompt = args.prompt
elif args.input_tokens in prompt_pool:
    prompt = prompt_pool[args.input_tokens]
else:
    raise SystemExit('[ERROR] Plese use --prompt if want to use custom input.')


# start
total_time = 0.0
num_iter = 10
num_warmup = 4


from intel_extension_for_transformers.backends.neural_engine.compile import compile
graph = compile(args.ir_path)
print("Using IR file {}".format(args.ir_path))
import numpy as np

prompt = [prompt] * args.batch_size
with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = model.generate(input_ids, max_new_tokens=args.max_new_tokens, engine_model = graph, **generate_kwargs)
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        toc = time.time()
        print(gen_text, flush=True)
        if i >= num_warmup:
            total_time += (toc - tic)

print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))

