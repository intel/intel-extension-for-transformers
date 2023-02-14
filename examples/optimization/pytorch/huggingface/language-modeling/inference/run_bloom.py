import argparse
import math
import time

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--benchmark", action="store_true", help="additionally run benchmark"
    )
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        help="bfloat16 or float32",
        choices=["float32", "bfloat16"],
        default="bfloat16",
    )

    return parser.parse_args()


t_start = time.time()

num_tokens = 32

args = get_args()

model_name = "bigscience/bloom"
print(f"Loading model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

kwargs = {
    "torch_dtype": getattr(torch, args.dtype),
    "low_cpu_mem_usage": True,
}

model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

if args.benchmark:
    t_ready = time.time()

### Generate

print(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

# 32 tokens
input_sentences = [
    "DeepSpeed is a machine learning framework for deep neural networks (DNNs) and deep"
    " reinforcement learning (DRL). It is written in C++ and is",
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

if args.greedy:
    generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)
else:
    generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False, num_beams=4)


print(f"Generate args {generate_kwargs}")
inputs = input_sentences[: args.batch_size]


def generate():
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(
        inputs, return_tensors="pt", padding=True
    )
    outputs = model.generate(**input_tokens, **generate_kwargs)
    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [
        o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
    ]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


print("*** Running generate")
t_generate_start = time.time()
generated = generate()
t_generate_span = time.time() - t_generate_start
for i, o, _ in generated:
    print(f"{'-'*60}\nin={i}\nout={o}\n")

### Benchmark

if args.benchmark:

    print("*** Running benchmark")
    # warm up
    for i in range(1):
        _ = generate()

    # latency
    t0 = time.time()
    generated = generate()
    print(
        f"""
*** Performance stats:
latency of {args.batch_size} full sentence: {time.time() - t0:.3f} secs
"""
    )
    # throughput
    t0 = time.time()
    cycles = 5
    total_new_tokens_generated = 0
    for i in range(cycles):
        generated = generate()
        total_new_tokens_generated += sum(new_tokens for _, _, new_tokens in generated)

    throughput = (time.time() - t0) / (total_new_tokens_generated)
    print(
        f"""
*** Performance stats:
Throughput per token including tokenize: {throughput*1000:.2f} msecs with (bs={args.batch_size})
Start to ready to generate: {t_ready - t_start:.3f} secs.
"""
    )
