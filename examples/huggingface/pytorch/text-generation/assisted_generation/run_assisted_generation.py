import torch
import time
import argparse

# import logging
# logging.disable(logging.WARNING)

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)

MODEL_CLASSES = {
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
}

DEFAULT_ASSISTANT_MODELS = {
    "OPTForCausalLM": "facebook/opt-125m",
    "LlamaForCausalLM": "PY007/TinyLlama-1.1B-intermediate-step-715k-1.5T",
    "GPTBigCodeForCausalLM": "bigcode/tiny_starcoder_py"
}

# args
parser = argparse.ArgumentParser("Text generation script", add_help=False)
parser.add_argument(
    "-m", "--model-id", type=str, required=True, help="huggingface model")
parser.add_argument(
    "-n", "--no-assisted", default=False, action=argparse.BooleanOptionalAction, help='Disable assisted generation.')
parser.add_argument(
    "--assistant-model", type=str, default=None, help="Override the default assistant model")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
parser.add_argument("--max-new-tokens", default=128, type=int, help="max new tokens")

args = parser.parse_args()
print(args)

# device
device = torch.device("cpu")

# torch dtype
if args.dtype == "bfloat16":
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32

def load_model(model_id):
    model_type = next((x for x in MODEL_CLASSES.keys() if x in model_id.lower()), 'auto')
    model_class = MODEL_CLASSES[model_type]
    print("Load model via", model_class)
    model = model_class[0].from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=amp_dtype)
    print("Model dtype:", model.config.torch_dtype)
    model = model.eval().to(device)
    model = model.to(memory_format=torch.channels_last)
    return model, model_class

model, model_class = load_model(args.model_id)
tokenizer = model_class[1].from_pretrained(args.model_id)

if args.no_assisted:
    assistant_model = None
else:
    if args.assistant_model is not None:
        assistant_model_name = args.assistant_model
    else:
        assistant_model_name = DEFAULT_ASSISTANT_MODELS.get(model.config.architectures[0])
        if assistant_model_name is None:
            raise ValueError(f"No default assistant model associated with {args.model_id}, please specify using `--assistant_model`.")

    print(f"Assistant model: {assistant_model_name}")
    assistant_model, _ = load_model(assistant_model_name)

# generate args
generate_kwargs = dict(
    do_sample=False, max_new_tokens=args.max_new_tokens, assistant_model=assistant_model
)
text_generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, **generate_kwargs
)

# input prompt
if "code" in model.config.architectures[0].lower():
    prompt = """def string_sequence(n: int) -> str: \"\"\" Return a string containing space-delimited numbers starting from 0 upto n inclusive. >>> string_sequence(0) '0' >>> string_sequence(5) '0 1 2 3 4 5' \"\"\""""
else:
    prompt = "DeepSpeed is a machine learning framework for deep neural networks and deep reinforcement learning. It is written in C++ and is available for Linux, Mac OS X,"
prompt = [prompt]
input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)

# start
num_iter = 10
num_warmup = 3


def torch_generation():
    total_time = 0.0
    with torch.inference_mode(), torch.autocast(
        "cpu", enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None
    ):
        for i in range(num_iter):
            tic = time.perf_counter()
            output = text_generator(prompt)
            toc = time.perf_counter()
            print(output, flush=True)
            if i >= num_warmup:
                total_time += toc - tic
    return total_time, num_iter - num_warmup


total_time, total_iter = torch_generation()
latency = total_time / total_iter
print(
    "\nSentence latency: %.3f sec, Token latency: %.3f sec, Token throughput: %.2f tokens/s.\n"
    % (latency, latency / args.max_new_tokens, args.max_new_tokens / latency)
)
