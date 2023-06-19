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
from optimum.utils import NormalizedConfigManager

# args
parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
parser.add_argument("--model_path",
        type=str,
        help="path to bfloat16 or int8 IR files",
        default="bfloat16",
    )
parser.add_argument("--model",
        type=str,
        help="path to original config and weight files",
        default="EleutherAI/gpt-j-6B",
    )
parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")
parser.add_argument('--input-tokens', default='32', type=str)
parser.add_argument('--prompt', default=None, type=str)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--weight_type', default=None, type=str)
args = parser.parse_args()
print(args)

generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

model_id = args.model
config = AutoConfig.from_pretrained(model_id)
model_type = config.model_type
normalized_config = NormalizedConfigManager.get_normalized_config_class(model_type)(config)
num_attention_heads = normalized_config.num_attention_heads
hidden_size = normalized_config.hidden_size
generate_kwargs["past_kv_nums"] = normalized_config.num_layers
generate_kwargs["model_type"] = model_type
generate_kwargs["num_attention_heads"] = num_attention_heads
generate_kwargs["d_k"] = hidden_size // num_attention_heads
generate_kwargs["vocab_size"] = normalized_config.vocab_size

if 'llama' in model_type:
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    prompt_json = '/llamaprompt.json'
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt_json = '/prompt.json'

# load model
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
with open(str(current_path) + prompt_json) as f:
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


from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
if args.weight_type:
    with autocast('bf16', weight_dtype=args.weight_type):
        print("Using FP8 weight which has storage type {} and make sure your IR is BF16 type".format(
              args.weight_type))
        graph = compile(args.model_path)
else:
    graph = compile(args.model_path)
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

