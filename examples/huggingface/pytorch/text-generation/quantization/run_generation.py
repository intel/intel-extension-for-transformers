import argparse
import re
import time
import json
import torch
import logging
from transformers import AutoConfig, AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from transformers.utils import check_min_version
from optimum.intel.generation.modeling import TSModelForCausalLM
from intel_extension_for_transformers.transformers import (
    MixedPrecisionConfig,
    WeightOnlyQuantConfig,
    SmoothQuantConfig,
    BitsAndBytesConfig

)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6B", const="EleutherAI/gpt-j-6B"
)
parser.add_argument("--revision", default=None, type=str)
parser.add_argument("--trust_remote_code", default=False)
parser.add_argument(
    "--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")
# ============Benchmark configs==============
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=56, type=int,
                    help="batch size num.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai"], type=str, \
                    help="tasks list for accuracy validation")
# ============MixedPrecision configs==============
parser.add_argument("--mixed_precision", action="store_true")
# ============SmoothQuant configs==============
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default="0.5", help="Smooth quant parameter.")
# ============WeightOnlyQuant configs===============
parser.add_argument("--woq", action="store_true")
parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ'], 
                    help="Weight-only parameter.")
parser.add_argument("--woq_dtype", type=str, default="int8", 
                    choices=["int8", "int4_clip", "int4_fullrange", "fp4_e2m1_bnb", "fp4_e2m1", "nf4"])
parser.add_argument("--woq_group_size", type=int, default=-1)
parser.add_argument("--woq_scheme", default="sym")
parser.add_argument("--woq_enable_mse_search", action="store_true")
parser.add_argument("--woq_enable_full_range", action="store_true")
# ============BitsAndBytes configs==============
parser.add_argument("--bitsandbytes", action="store_true")
parser.add_argument("--load_in_4bit", type=bool, default=False)
parser.add_argument("--load_in_8bit", type=bool, default=False)
# =======================================
args = parser.parse_args()

# transformers version >= 4.32.0 contained the mpt modeling definition.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py
check_min_version("4.32.0")

# get model config
config = AutoConfig.from_pretrained(
      args.model,
      torchscript=True
      if (args.sq or args.woq_algo in ['AWQ', 'TEQ'])
      else False,  # torchscript will force `return_dict=False` to avoid jit errors
      use_cache=True, # to use kv cache.
      trust_remote_code=args.trust_remote_code,
      revision=args.revision,
      )


# tokenizer
if config.model_type == "llama":
   from transformers import LlamaTokenizer
   tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
   tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

# quantization config setting
quantization_config = None
if args.mixed_precision:
    quantization_config = MixedPrecisionConfig(dtype="bfloat16") # default is bfloat16
elif args.sq:
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
    quantization_config = SmoothQuantConfig(
                                tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
                                alpha="auto" if args.alpha == "auto" else float(args.alpha),    # default is 0.5
                                op_type_dict=op_type_dict,  # default is {}
                                excluded_precisions=excluded_precisions,  # default is []
                               )
elif args.woq:
    quantization_config = WeightOnlyQuantConfig(compute_type="fp32", weight_type="int4_fullrange", group_size=32) #default is A32W4G32
# bitsandbytes
elif args.bitsandbytes:
    # GPU device is need for `load_in_4bit` and `load_in_8bit`.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

# get model
# `BitsAndBytesConfig` and (`load_in_4bit` or `load_in_8bit`) is alternative for WeightOnlyQuant.
if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                      quantization_config=quantization_config,
                                                      trust_remote_code=args.trust_remote_code,
                                                      use_llm_runtime=False
                                                      )
    if args.sq:
        config.save_pretrained(args.output_dir)
        user_model.save(args.output_dir)
elif args.load_in_4bit or args.load_in_8bit:
    # CPU device usage is provided by intel-extension-for-transformers.
    user_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                      load_in_4bit=args.load_in_4bit,
                                                      load_in_8bit=args.load_in_8bit,
                                                      use_llm_runtime=False
                                                      )
elif not args.int8 and not args.int8_bf16_mixed:
    user_model = AutoModelForCausalLM.from_pretrained(args.model, config=config, trust_remote_code=args.trust_remote_code, use_llm_runtime=False)
    # peft
    if args.peft_model_id is not None:
        from peft import PeftModel
        user_model = PeftModel.from_pretrained(user_model, args.peft_model_id)

# Generation
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)


if args.int8 or args.int8_bf16_mixed:
    # TorchScript model don't attribute generate method, the wrapper is provided.
    import intel_extension_for_pytorch as ipex
    user_model = TSModelForCausalLM.from_pretrained(
        args.output_dir, file_name="best_model.pt", trust_remote_code=args.trust_remote_code
    )


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

