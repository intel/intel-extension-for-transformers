import argparse
import os
import re
import time
import json
import torch
import logging
from transformers import AutoConfig, AutoTokenizer
from intel_extension_for_transformers.transformers import (
        AutoModelForCausalLM,
        AutoModel
)
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
    "--model",  default=None
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
parser.add_argument("--quantized_model_path", type=str, default="saved_results/best_model.pt", help="the int8 model path")
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
# 4.31.0 for ipex.optimize_transformers
check_min_version("4.31.0")

# get model config
if args.peft_model_id:
    from peft import PeftConfig
    peft_config = PeftConfig.from_pretrained(args.peft_model_id)
    if args.model is None:
        args.model = peft_config.base_model_name_or_path
        print("we will use peft base_model_name_or_path to get tokenizer.")

config = AutoConfig.from_pretrained(
    args.model,
    torchscript=True
    if (args.sq or args.woq_algo in ['AWQ', 'TEQ'] or (args.int8 or args.int8_bf16_mixed))
    else False,  # torchscript will force `return_dict=False` to avoid jit errors
    use_cache=True, # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    revision=args.revision,
    )

# chatglm
if config.model_type == "chatglm":
    AutoModelForCausalLM = AutoModel
# tokenizer
if config.model_type == "llama":
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

# use peft
args.model = args.peft_model_id if args.peft_model_id is not None else args.model

# Generation
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# mp/sq/woq/bitsandbytes config setting
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
            "<built-in function linear>": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        }
    elif re.search("mistral", config.model_type) or re.search("baichuan", config.model_type):
        op_type_dict = {".*": {"activation": {"algorithm": "minmax"}}}
    else:
        op_type_dict = {}
    if re.search("dolly", args.model):
        ipex_opt_llm = False
    else:
        ipex_opt_llm = None
    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    recipes = {
                "smooth_quant": True,
                "smooth_quant_args": {"alpha": args.alpha},
            }
    quantization_config = SmoothQuantConfig(
        tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
        recipes=recipes,
        op_type_dict=op_type_dict,  # default is {}
        excluded_precisions=excluded_precisions,  # default is []
        num_beams=generate_kwargs["num_beams"],
        ipex_opt_llm=ipex_opt_llm
        )
elif args.woq:
    quantization_config = WeightOnlyQuantConfig(compute_dtype="fp32", weight_dtype="int4_fullrange", group_size=32) #default is A32W4G32
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
    # save model
    if args.sq:
        config.save_pretrained(args.output_dir)
        user_model.save(args.output_dir)
    elif args.mixed_precision:
        user_model.config.save_pretrained(args.output_dir)
        torch.save(user_model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))

elif args.load_in_4bit or args.load_in_8bit:
    # CPU device usage is provided by intel-extension-for-transformers.
    user_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                      load_in_4bit=args.load_in_4bit,
                                                      load_in_8bit=args.load_in_8bit,
                                                      use_llm_runtime=False
                                                      )
elif not args.int8 and not args.int8_bf16_mixed:
    if args.peft_model_id is not None:
        user_model = AutoModelForCausalLM.from_pretrained(args.peft_model_id, trust_remote_code=args.trust_remote_code, use_llm_runtime=False)
    else:
        user_model = AutoModelForCausalLM.from_pretrained(args.model, config=config, trust_remote_code=args.trust_remote_code, use_llm_runtime=False)


if args.int8 or args.int8_bf16_mixed:
    # TorchScript model don't attribute generate method, the wrapper is provided.
    import intel_extension_for_pytorch as ipex
    if config.model_type in ["gptj", "opt", "llama", "gpt_neox"] and not re.search("dolly", args.model):
        if args.accuracy:
            from intel_extension_for_transformers.llm.evaluation.models import TSModelCausalLMForOPTLLM
            user_model = TSModelCausalLMForOPTLLM.from_pretrained(
                args.output_dir, file_name="best_model.pt", trust_remote_code=args.trust_remote_code
            )
        else:
            torch._C._jit_set_texpr_fuser_enabled(False)
            qconfig = ipex.quantization.default_static_qconfig_mapping
            with ipex.OnDevice(dtype=torch.float, device="meta"):
                user_model = AutoModelForCausalLM.from_pretrained(args.model, use_llm_runtime=False)
            user_model = ipex.optimize_transformers(
                user_model.eval(),
                dtype=torch.float,
                inplace=True,
                quantization_config=qconfig,
                deployment_mode=False,
            )
            if not hasattr(user_model, "trace_graph"):
                print("load_quantized_model")
                self_jit = torch.jit.load(args.quantized_model_path)
                self_jit = torch.jit.freeze(self_jit.eval())
                ipex._set_optimized_model_for_generation(user_model, optimized_model=self_jit)            
    else:
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
    args.model = peft_config.base_model_name_or_path if args.peft_model_id else args.model
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    results = evaluate(
        model="hf-causal",
        model_args='pretrained=' + args.model + ',tokenizer=' + args.model + \
            ',dtype=float32' + ",trust_remote_code=" + str(args.trust_remote_code),
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
