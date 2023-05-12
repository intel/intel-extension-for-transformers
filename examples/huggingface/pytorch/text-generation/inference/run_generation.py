import torch
import time
import argparse

# import logging
# logging.disable(logging.WARNING)

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoModel,
    LlamaForCausalLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaTokenizer,
)

MODEL_CLASSES = {
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "chatglm": (AutoModel, AutoTokenizer),
}

# args
parser = argparse.ArgumentParser("Text generation script", add_help=False)
parser.add_argument(
    "-m", "--model-id", type=str, required=True, help="huggingface model"
)
parser.add_argument("--dtype", type=str, default="float32", help="bfloat16 or float32")
parser.add_argument("--max-new-tokens", default=32, type=int, help="max new tokens")
parser.add_argument("--ipex", action="store_true", help="Use IPEX")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
args = parser.parse_args()
print(args)


# device
device = torch.device("cpu")

if args.ipex:
    from optimum.intel import inference_mode as ipex_inference_mode

    torch._C._jit_set_texpr_fuser_enabled(False)

# torch dtype
if args.dtype == "bfloat16":
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32

# load model
model_type = next((x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), 'auto')
model_class = MODEL_CLASSES[model_type]
print("Load model via", model_class)
model = model_class[0].from_pretrained(
    args.model_id, low_cpu_mem_usage=True, return_dict=not args.ipex, torch_dtype=amp_dtype
)
print("Model dtype:", model.config.torch_dtype)
tokenizer = model_class[1].from_pretrained(args.model_id)
model = model.eval().to(device)
model = model.to(memory_format=torch.channels_last)

# generate args
generate_kwargs = dict(
    do_sample=False, temperature=0.9, num_beams=4, max_new_tokens=args.max_new_tokens
)
text_generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, **generate_kwargs
)

# input prompt
prompt = "DeepSpeed is a machine learning framework for deep neural networks and deep reinforcement learning. It is written in C++ and is available for Linux, Mac OS X,"
prompt = [prompt] * args.batch_size
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
            tic = time.time()
            output = text_generator(prompt)
            toc = time.time()
            print(output, flush=True)
            if i >= num_warmup:
                total_time += toc - tic
    return total_time, num_iter - num_warmup


def ipex_generation():
    total_time = 0.0
    with ipex_inference_mode(
        text_generator, dtype=amp_dtype, jit=True
    ) as ipex_text_generator:
        for i in range(num_iter):
            tic = time.time()
            output = ipex_text_generator(prompt)
            toc = time.time()
            print(output, flush=True)
            if i >= num_warmup:
                total_time += toc - tic
    return total_time, num_iter - num_warmup


total_time, total_iter = ipex_generation() if args.ipex else torch_generation()
latency = total_time / total_iter
print(
    "\nSentence latency: %.3f sec, Token latency: %.3f sec, Token throughput: %.2f tokens/s.\n"
    % (latency, latency / args.max_new_tokens, args.max_new_tokens / latency)
)
