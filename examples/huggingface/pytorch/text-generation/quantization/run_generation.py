import argparse
import re
import time
import json
import os
import pathlib
import torch
import types
from pathlib import Path
from datasets import load_dataset, load_from_disk
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.utils import check_min_version
import transformers
import numpy as np
from itertools import chain
from optimum.utils import NormalizedConfigManager
from optimum.intel.generation.modeling import TSModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6B", const="EleutherAI/gpt-j-6B"
)
parser.add_argument("--revision", default=None, type=str)
parser.add_argument("--trust_remote_code", default=False)
parser.add_argument(
    "--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
)
parser.add_argument("--dtype", type=str, default="int8")
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--alpha", default="auto", help="Smooth quant parameter.")
parser.add_argument(
    "--pad_max_length", default=512, type=int, help="Pad input ids to max length."
)
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="batch size num.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--tasks", nargs='+', default=["winogrande", "copa", "piqa", "rte", "hellaswag", \
                    "openbookqa", "lambada_openai", "lambada_standard", "wikitext"], type=str, \
                    help="tasks list for accuracy validation")

args = parser.parse_args()

calib_size = 1

# model
config = AutoConfig.from_pretrained(
      args.model,
      torchscript=True
      if args.quantize
      else False,  # torchscript will force `return_dict=False` to avoid jit errors
      use_cache=True, # to use kv cache.
      trust_remote_code=args.trust_remote_code,
      revision=args.revision,
      )

# transformers version >= 4.32.0 contained the mpt modeling definition.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py
if config.model_type == "mpt":
    check_min_version("4.32.0")

# tokenizer
if config.model_type == "llama":
   from transformers import LlamaTokenizer
   tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
   tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

# quantize
if args.quantize:
    from intel_extension_for_transformers.transformers import (
        AMPConfig,
        WeightOnlyQuantizationConfig,
        SmoothQuantConfig,
        BitsAndBytesConfig

    ) 
    from intel_extension_for_transformers.transformers import AutoModelForCausalLM
    if re.search("gptj", config.model_type) or re.search(
        "gpt_neox", config.model_type
    ):
        op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        }
    elif re.search("mpt", config.model_type):
        op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
            "<built-in function linear>":{"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        }
    else:
        op_type_dict = {}
    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    sq_config = SmoothQuantConfig(
                                tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
                                alpha=float(args.alpha),    # default is 0.5
                                op_type_dict=op_type_dict,  # default is {}
                                excluded_precisions=excluded_precisions,  # default is []
                               )
    # smooth-quant
    q_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                   quantization_config=sq_config
                                               )
    print("sq done.")
    # weight-only
    woq_config = WeightOnlyQuantizationConfig(algorithm="RTN", # default is "RTN"
                                              bits=8, # default is 8
                                              group_size=-1, # default is -1
                                              scheme="sym", # default is sym
                                              enable_full_range=True # default is True
                                              ) 
    woq_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                quantization_config=woq_config
                                            )
    print("woq done.")
    # amp
    amp_config = AMPConfig(dtype="bfloat16") # default is bfloat16
    amp_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                quantization_config=amp_config
                                            )
    print("amp done.")
    # bitsandbytes
    bab_config = BitsAndBytesConfig()
    bab_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                quantization_config=bab_config
                                            )
    print("bitsandbytes done.")


# Generation
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

if args.int8 or args.int8_bf16_mixed:
    # TorchScript model don't attribute generate method, the wrapper is provided.
    if args.ipex:
        user_model = TSModelForCausalLM.from_pretrained(
            args.output_dir, file_name="best_model.pt", trust_remote_code=args.trust_remote_code
        )
    else:
        from neural_compressor.utils.pytorch import load

        user_model = load(args.output_dir, user_model)


if args.benchmark:
    prompt = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    # start
    total_time = 0.0
    num_iter = args.iters
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    total_token_num = 0

    with torch.inference_mode(), torch.no_grad():
        for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            gen_ids = user_model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
            )
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time.time()
            # please check the gen_ids if include input_ids.
            input_tokens_num = input_ids.numel()
            output_tokens_num = gen_ids.numel() - input_tokens_num
            print(gen_text, flush=True)
            if i >= num_warmup:
                total_time += toc - tic
                total_token_num += output_tokens_num

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / total_token_num
    print("Inference latency: %.3f sec." % latency)
    throughput = total_token_num / total_time
    print("Throughput: {} samples/sec".format(throughput))

if args.accuracy:
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    results = evaluate(
        model="hf-causal",
        model_args='pretrained='+args.model+',tokenizer='+args.model+',dtype=float32',
        user_model=user_model,
        batch_size=args.batch_size,
        tasks=args.tasks,
    )
    dumped = json.dumps(results, indent=2)
    if args.save_accuracy_path:
        with open(args.save_accuracy_path, "w") as f:
            f.write(dumped)
    for task_name in args.tasks:
        if task_name == "wikitext":
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]))
        else:
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]))
