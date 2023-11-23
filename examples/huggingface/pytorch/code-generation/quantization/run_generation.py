import argparse
import re
import time
import json
import os
import pathlib
import torch
import types
import numpy as np
from itertools import chain
from pathlib import Path
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PretrainedConfig, AutoConfig
import transformers
from optimum.utils import NormalizedConfigManager
from intel_extension_for_transformers.transformers import (
    WeightOnlyQuantConfig,
    SmoothQuantConfig,
)
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
    AutoModel
)

parser = argparse.ArgumentParser()

# Main config
parser.add_argument(
    "--model", nargs="?", default="bigcode/starcoderbase", const="bigcode/starcoderbase"
)
parser.add_argument("--trust_remote_code", default=False)
parser.add_argument("--revision", default=None, type=str)
parser.add_argument(
    "--dataset", nargs="?", default="mbpp", const="mbpp"
)
parser.add_argument("--dtype", type=str, default="int8")
parser.add_argument(
    "--max_new_tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default="0.5", help="Smooth quant parameter.")
parser.add_argument(
    "--pad_max_length", default=512, type=int, help="Pad input ids to max length."
)
parser.add_argument("--calib_iters", default=32, type=int, help="calibration iters.")
parser.add_argument("--calib_batch_size", default=1, type=int, help="calibration batch size.")
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
parser.add_argument("--prompt_size", default=32, type=int, help="generate dummy input_ids size")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="batch size num.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--tasks", default="humaneval", type=str, \
                    help="tasks list for accuracy validation")
# WeightOnlyQuant config
parser.add_argument("--woq", action="store_true")
parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ'], 
                    help="Weight-only parameter.")
parser.add_argument("--woq_dtype", type=str, default="int4_fullrange", 
                    choices=["int8", "int4_clip", "int4_fullrange", "fp4_e2m1_bnb", "fp4_e2m1", "nf4"])
parser.add_argument("--woq_group_size", type=int, default=32)
parser.add_argument("--woq_scheme", default="sym")
parser.add_argument("--woq_enable_mse_search", action="store_true")
parser.add_argument("--woq_enable_full_range", action="store_true")
# Harness config
parser.add_argument("--n_samples", default=200, type=int)
parser.add_argument("--limit", default=None, type=int, help="Limit number of samples to eval")
parser.add_argument("--allow_code_execution", action="store_true")
#parser.add_argument("--precision", default="fp32")
parser.add_argument("--prefix", default="")
parser.add_argument("--generation_only", action="store_true")
parser.add_argument("--postprocess", action="store_false")
parser.add_argument("--save_references", action="store_true")
parser.add_argument("--save_generations", action="store_true")
parser.add_argument("--instruction_tokens", default=None)
parser.add_argument("--save_generations_path", default="generations.json")
parser.add_argument("--load_generations_path", default=None)
parser.add_argument("--metric_output_path", default="evaluation_results.json")
parser.add_argument("--seed", default=0, type=int)
# Generation config
parser.add_argument("--max_length_generation", default=512, type=int)
parser.add_argument("--temperature", default=0.8, type=float)
parser.add_argument("--top_p", default=0.8, type=float)
parser.add_argument("--top_k", default=0, type=int)
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--check_references", action="store_true")
parser.add_argument("--max_memory_per_gpu", type=str, default=None)
parser.add_argument(
    "--modeltype",
    default="causal",
    help="AutoModel to use, it can be causal or seq2seq",
)
parser.add_argument(
    "--limit_start",
    type=int,
    default=0,
    help="Optional offset to start from when limiting the number of samples",
)   
args = parser.parse_args()

from intel_extension_for_transformers.transformers import AutoModelForCausalLM
user_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torchscript=True
    if args.ipex
    else False,  # torchscript will force `return_dict=False` to avoid jit errors
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    # revision=args.revision,
    # trust_remote_code=args.trust_remote_code,
    # use_auth_token=args.use_auth_token,
    truncation_side="left",
    padding_side="right",
)
config = AutoConfig.from_pretrained(
    args.model,
    torchscript=True
    if (args.sq or args.woq_algo in ['AWQ', 'TEQ'] or (args.int8 or args.int8_bf16_mixed))
    else False,  # torchscript will force `return_dict=False` to avoid jit errors
    use_cache=True, # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    revision=args.revision,
)
if not tokenizer.eos_token:
    if tokenizer.bos_token:
        tokenizer.eos_token = tokenizer.bos_token
        print("bos_token used as eos_token")
    else:
        raise ValueError("No eos_token or bos_token found")

tokenizer.pad_token = tokenizer.eos_token

# to channels last
user_model = user_model.to(memory_format=torch.channels_last)
user_model.eval()

if args.ipex:
    import intel_extension_for_pytorch as ipex
    from optimum.intel.generation.modeling import TSModelForCausalLM

calib_dataset = args.dataset
op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        }
excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
quantization_config = None
# sq/woq config setting
if args.sq:
    quantization_config = SmoothQuantConfig(
        tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
        alpha="auto" if args.alpha == "auto" else float(args.alpha),    # default is 0.5
        op_type_dict=op_type_dict,  # default is {}
        excluded_precisions=excluded_precisions,  # default is []
        calib_dataset=calib_dataset,
        calib_iters=args.calib_iters
    )
elif args.woq:
    quantization_config = WeightOnlyQuantConfig(
        weight_dtype=args.woq_dtype,
        group_size=args.woq_group_size,
        scheme=args.woq_scheme,
        algorithm=args.woq_algo
    ) #default is A32W4G32
else:
    from neural_compressor import PostTrainingQuantConfig
    quantization_config = PostTrainingQuantConfig(
        backend="ipex" if args.ipex else "default",
        excluded_precisions=excluded_precisions,
        op_type_dict=op_type_dict,
        calib_dataset=calib_dataset,
        calib_iters=args.calib_iters
    )

if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
        use_llm_runtime=False
    )
    # save model
    if args.sq:
        config.save_pretrained(args.output_dir)
        user_model.save(args.output_dir)

# Generation
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

if args.int8 or args.int8_bf16_mixed:
    if args.ipex:
        # TorchScript model don't attribute generate method, the wrapper is provided.
        import intel_extension_for_pytorch as ipex
        from intel_extension_for_transformers.llm.evaluation?models import TSModelCausalLMForOPTLLM
        user_model = TSModelCausalLMForOPTLLM.from_pretrained(
            args.output_dir, file_name="best_model.pt", trust_remote_code=args.trust_remote_code
        )
        print("Load torchscript int8 model successfully.")
    else:
        from neural_compressor.utils.pytorch import load
        user_model = load(args.output_dir, user_model)
        print("Load int8 model successfully.")
    
if args.benchmark:
    print("---- Prompt size:", args.prompt_size)

    normalized_config = NormalizedConfigManager.get_normalized_config_class(
            user_model.config.model_type
        )(user_model.config)

    num_layers = normalized_config.num_layers
    num_attention_heads = normalized_config.num_attention_heads
    hidden_size = normalized_config.hidden_size
    d_k = hidden_size // num_attention_heads

    num_iter = args.iters
    num_warmup = args.num_warmup

    total_latency = 0
    for j in range(args.max_new_tokens):
        total_time = 0.0
        with torch.inference_mode(), torch.no_grad():

            for i in range(num_iter):
                tic = time.time()
                if j==0:
                    #input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    input_ids = torch.randint(1, tokenizer.vocab_size, size = (args.batch_size , args.prompt_size))
                    attention_mask = torch.ones(input_ids.shape)
                    new_shape = [input_ids.shape[0], 0, d_k*2]
                    dummy_tensor = torch.ones(size=new_shape)
                    past_key_values = tuple([dummy_tensor] * num_layers)

                inp = {"input_ids": input_ids,
                        "past_key_values": past_key_values,
                        "attention_mask": attention_mask}

                out = user_model(**inp)
                gen_id = torch.argmax(out[0][:, -1:, :], axis = -1)
                gen_text = tokenizer.batch_decode(gen_id, skip_special_tokens=True)
                toc = time.time()
                #print(gen_text, flush=True)
                if i >= num_warmup:
                    total_time += toc - tic

        print("\n", "-" * 10, "Summary:", "-" * 10)
        print("Generated token index:", j+1)
        latency = total_time / (num_iter - num_warmup)
        print("Inference latency: %.5f sec." % latency)
        throughput = (num_iter - num_warmup) / total_time
        print("Throughput: {} samples/sec".format(throughput))

        input_ids = gen_id
        past_key_values = out[1]
        attention_mask = torch.ones((attention_mask.shape[0], attention_mask.shape[1] + 1))
        total_latency += latency

    average_latency = total_latency / args.max_new_tokens
    print("Average inference latency: %.5f sec." % latency)
    average_throughput = args.max_new_tokens / total_latency
    print("Average throughput: {} samples/sec".format(throughput))


if args.accuracy:
    from intel_extension_for_transformers.llm.evaluation.lm_code_eval import evaluate
    results = evaluate(
        model=user_model,
        tokenizer=tokenizer,
        tasks=args.tasks,
        batch_size=args.batch_size,
        args=args,
    )

    print(results)
