"""
Conditional text generation on Habana Gaudi/Gaudi2.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from itertools import cycle
from pathlib import Path

import torch
import datasets
from utils import adjust_batch, count_hpu_graphs, initialize_model
from utils import print_memory_stats, compute_perplexity

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_parser(parser):
    # Arguments management
    parser.add_argument("--device", "-d", type=str, choices=["hpu"], help="Device to run", default="hpu")
    parser.add_argument(
            "--dataset_name", 
            type=str, choices=["tau/scrolls", "hoskinson-center/proof-pile", "pg19"],
            help="Dataset to use", default=None)
    parser.add_argument(
            "--tokenized", 
            type=str, help="Dataset has been tokenized", default=None)
    parser.add_argument(
            "--save_tokenized", 
            type=str, help="Path to save tokenized file", default=None)
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument("--size", type=int, default=1, help="Enlarge the input prompt")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        nargs="*",
        help='Optional argument to give a prompt of your choice as input. Can be a single string (eg: --prompt "Hello world"), or a list of space-separated strings (eg: --prompt "Hello world" "How are you?")',
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--limit_hpu_graphs",
        action="store_true",
        help="Skip HPU Graph usage for first token to save memory",
    )
    parser.add_argument("--fp8", action="store_true", help="Enable Quantization to fp8")
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to use torch compiled model or not.",
    )
    args = parser.parse_args()
    #(TODO) we will use kv-cache in cpu side so we do not use hpu graphs
    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    return args

def main():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)
    model, tokenizer, generation_config = initialize_model(args)
    # print_memory_stats()
    use_lazy_mode = True
    if args.torch_compile and model.config.model_type == "llama":
        use_lazy_mode = False

    import habana_frameworks.torch.hpu as torch_hpu

    # Benchmark over the prompts below
    if args.dataset_name:
        if args.tokenized:
            input_sentences = datasets.load_from_disk(args.tokenized)
        else:
            input_sentences = datasets.load_dataset(args.dataset_name, split='test')
            tokenizer.pad_token = tokenizer.eos_token
            feature = 'text'
            def tokenize(example):
                tokenized = tokenizer(
                    example[feature],
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                    max_length=sys.maxsize,
                    return_attention_mask=True,
                )
                example["input_ids"] = tokenized["input_ids"]
                example["attention_mask"] = tokenized["attention_mask"]
                example["tokenized_len"] = len(tokenized["input_ids"])
                return example
    
            input_sentences = input_sentences.map(tokenize)
    
        input_sentences = input_sentences.filter(lambda x: x["tokenized_len"] >= args.size * 1024)
        input_sentences = input_sentences.filter(lambda x: x["tokenized_len"] <= (args.size + 2) * 1024)
        # TODO you can save the tokenized results, this will save a lot of time
        if args.save_tokenized:
            input_sentences.save_to_disk(args.save_tokenized)
            print("Token has been save to {}".format(args.save_tokenized))
            return

    else:
        input_sentences = [p * args.size for p in args.prompt]
        # (TODO) if we want to test multi-batch use this code
        # input_sentences = [
        #     "DeepSpeed is a machine learning framework",
        #     "He is working on",
        #     "He has a",
        #     "He got all",
        #     "Everyone is happy and I can",
        #     "The new movie that got Oscar this year",
        #     "In the far far distance from our galaxy,",
        #     "Peace is the only way",
        # ]

        if args.batch_size > len(input_sentences):
            # Dynamically extends to support larger batch sizes
            num_sentences_to_add = args.batch_size - len(input_sentences)
            for i in range(num_sentences_to_add):
                input_sentences.append(input_sentences[i % len(input_sentences)])
        elif args.batch_size < len(input_sentences):
            input_sentences = input_sentences[: args.batch_size]

    def compute_ppl(inputs):
        results = []
        result = []
        has_bos_token = tokenizer.bos_token is not None
        # samples_number will influence the ppl
        samples_numbers = [50]
        sliding_windows = [256, 512, 1024]
        for samples_num  in samples_numbers:
            # for sliding_window in sliding_windows:
            ppl = compute_perplexity(
                    model, tokenizer, inputs, samples_num,
                    add_start_token=has_bos_token, max_length=args.size*1024,
                    sliding_window=512, truncate=True)
            print("PPL result is {}".format(ppl)) 
            result.append(ppl)
    
        result.insert(0, model)
        results.append(result)
        return results

    def generate(inputs, size=None):
        """Generates sequences from the input sentences and returns them."""
        # Tokenization
        if args.dataset_name:
            input_tokens = {'input_ids': torch.tensor([inputs[0]['input_ids']]),
                    'attention_mask': torch.tensor([inputs[0]['attention_mask']])}
        elif args.max_input_tokens > 0:
            input_tokens = tokenizer.batch_encode_plus(
                inputs,
                return_tensors="pt",
                padding="max_length",
                max_length=args.max_input_tokens,
                truncation=True,
            )
        else:
            input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)

        if size is not None:
            input_tokens = adjust_batch(input_tokens, size)
        # Move inputs to target device(s)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(args.device)
        outputs = model.generate(
            **input_tokens,
            generation_config=generation_config,
            lazy_mode=use_lazy_mode,
            hpu_graphs=args.use_hpu_graphs,
            profiling_steps=args.profiling_steps,
            profiling_warmup_steps=args.profiling_warmup_steps,
        ).cpu()
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    from optimum.habana.utils import HabanaProfile

    # compilation stage disable profiling
    HabanaProfile.disable()
    # Compilation
    logger.info("Graph compilation...")
    t0 = time.perf_counter()
    # The first three iterations take longer because of graph compilation
    for _ in range(args.warmup):
        warm_generated = generate(input_sentences, None)
    torch_hpu.synchronize()
    # print_memory_stats()
    compilation_duration = time.perf_counter() - t0

    HabanaProfile.enable()
    total_new_tokens_generated = 0
    logger.info("Running generate...")
    t0 = time.perf_counter()
    print(f"Graph compilation duration          = {compilation_duration} seconds")
    if args.dataset_name:
        ppl_results = compute_ppl(input_sentences)
    else:
        generated = generate(input_sentences, None)
    duration = time.perf_counter() - t0
    total_new_tokens_generated = args.batch_size * args.max_new_tokens
    throughput = total_new_tokens_generated / duration

    # (TODO) only open this when to check the accuracy of the output
    # if not args.dataset_name:
    #     for i, input_sentence in enumerate(zip(input_sentences)):
    #         print(f"input {i+1}: {input_sentence}\noutput {i+1}: {generated[i]}")

    stats = f"Throughput (including tokenization) = {throughput} tokens/second"
    stats = stats + f"\nNumber of HPU graphs                = {count_hpu_graphs()}"
    separator = "-" * 90
    print(separator)
    print("".format(duration))
    print("The input token size is {}K ".format(args.size))
    print(stats)
    print_memory_stats()
    print(separator)
    if args.quant_config:
        import habana_quantization_toolkit
        habana_quantization_toolkit.finish_measurements(model)

if __name__ == "__main__":
    main()
