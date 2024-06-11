import argparse
import json
import os
import re
import time

import torch
from optimum.intel.generation.modeling import TSModelForCausalLM
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import check_min_version

from intel_extension_for_transformers.transformers import (
    AutoModel,
    AutoModelForCausalLM,
    MixedPrecisionConfig,
    SmoothQuantConfig,
)
from intel_extension_for_transformers.transformers.utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None)
parser.add_argument(
    "--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
)
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--max_new_tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument(
    "--restore_sq_model_from_json",
    action="store_true",
    help="restore ipex quantized model from output_dir/best_configure.json",
)
parser.add_argument(
    "--peft_model_id", type=str, default=None, help="model_name_or_path of peft model"
)
# ============Benchmark configs==============
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=56, type=int, help="batch size num.")
parser.add_argument(
    "--tasks",
    default="lambada_openai",
    type=str,
    help="tasks list for accuracy validation",
)
# ============MixedPrecision configs==============
parser.add_argument("--mixed_precision", action="store_true")
# ============SmoothQuant configs==============
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default=0.5, help="Smooth quant parameter.")
parser.add_argument(
    "--n_samples", default=100, type=int, help="Smooth quant calibration samples."
)
parser.add_argument(
    "--seq_len", default=512, type=int, help="Smooth quant calibration input length."
)
# sq alpha "auto" parameters
parser.add_argument("--scale_sharing", action="store_true")
parser.add_argument(
    "--init_alpha", default=0.5, type=float, help="Smooth quant parameter."
)
parser.add_argument(
    "--alpha_min", default=0.0, type=float, help="Smooth quant parameter."
)
parser.add_argument(
    "--alpha_max", default=1.0, type=float, help="Smooth quant parameter."
)
parser.add_argument(
    "--alpha_step", default=0.1, type=float, help="Smooth quant parameter."
)
parser.add_argument("--shared_criterion", default="max", type=str)
parser.add_argument("--do_blockwise", action="store_true")
# ============AutoModel parameters==============
parser.add_argument("--_commit_hash", default=None, type=str)
parser.add_argument("--trust_remote_code", action="store_true")
# =======================================
args = parser.parse_args()
# transformers version >= 4.32.0 contained the mpt modeling definition.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py
# 4.31.0 for ipex.optimize_transformers
check_min_version("4.35.2")
# get model config
if args.peft_model_id:
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(args.peft_model_id)
    if args.model is None:
        args.model = peft_config.base_model_name_or_path
        print("we will use peft base_model_name_or_path to get tokenizer.")

config = AutoConfig.from_pretrained(
    args.model,
    torchscript=(
        True if args.sq else False
    ),  # torchscript will force `return_dict=False` to avoid jit errors
    use_cache=True,  # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    _commit_hash=args._commit_hash,
)

# chatglm
if config.model_type == "chatglm":
    AutoModelForCausalLM = AutoModel
# tokenizer
if hasattr(config, "auto_map") and "chatglm2" in config.auto_map["AutoConfig"]:
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm2-6b", trust_remote_code=True
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

# use peft
args.model = args.peft_model_id if args.peft_model_id is not None else args.model

# Generation
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# mp/sq/woq/bitsandbytes config setting
quantization_config = None
if args.mixed_precision:
    quantization_config = MixedPrecisionConfig(dtype="bfloat16")  # default is bfloat16
elif args.sq:
    excluded_precisions = ["bf16"]
    quantization_config = SmoothQuantConfig(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
        excluded_precisions=excluded_precisions,
        alpha=args.alpha if args.alpha == "auto" else float(args.alpha),
        scale_sharing=args.scale_sharing,
        init_alpha=args.init_alpha,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        shared_criterion=args.shared_criterion,
        do_blockwise=args.do_blockwise,
        num_beams=generate_kwargs["num_beams"],
    )
else:
    print("The quantization_config is None.")

# get optimized model
if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
    )
    # save model
    if args.output_dir is not None and (args.sq or args.mixed_precision):
        tokenizer.save_pretrained(args.output_dir)
        if args.sq:
            quantization_config.remove_redundant_parameters()
            config.quantization_config = quantization_config
            config.save_pretrained(args.output_dir)
            torch.jit.save(user_model, args.output_dir + "/pytorch_model.bin")
            with open(args.output_dir + "/best_configure.json", "w") as f:
                json.dump(user_model.tune_cfg, f, indent=4)
            # validate loading
            user_model = AutoModelForCausalLM.from_pretrained(
                args.output_dir,
                trust_remote_code=args.trust_remote_code,
                _commit_hash=args._commit_hash,
            )
        elif args.mixed_precision:
            user_model.save_pretrained(args.output_dir)

if args.restore_sq_model_from_json:
    from intel_extension_for_transformers.transformers.llm.quantization.sq_utils import (
        recover_model_from_json,
    )
<<<<<<< HEAD
    user_model = recover_model_from_json(
        args.model,
        os.path.join(args.output_dir, "best_configure.json"),
        args.trust_remote_code,
    )

=======

    if args.restore:
        from intel_extension_for_transformers.transformers.utils.utility import (
            recover_model_from_json,
        )
        user_model = recover_model_from_json(
            args.model,
            os.path.join(args.output_dir, "best_configure.json"),
            args.trust_remote_code,
        )
    else:
        user_model = torch.jit.load(os.path.join( args.model, "best_model.pt"))
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        origin_model_type = config.model_type
        if origin_model_type in ["chatglm", "qwen", "baichuan"]:
            config.model_type = "qwen2"
        user_model = TSModelCausalLMForITREX(user_model, config=config)
        user_model.config.model_type = origin_model_type
>>>>>>> main
elif not (args.sq or args.mixed_precision):
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
    )


if args.benchmark:
    user_model = user_model.eval() if hasattr(user_model, "eval") else user_model
    prompt = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."
    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    # start
    total_time = 0.0
    num_iter = args.iters
    num_warmup = args.num_warmup
    total_token_num = 0
    eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode(), torch.no_grad():
        for i in range(num_iter):
            tic = time.time()
            # for chatglm2 only
            if hasattr(tokenizer, "build_chat_input"):
                input_ids = tokenizer.build_chat_input(prompt)["input_ids"]
                input_ids = input_ids.repeat(args.batch_size, 1)
                eos_token_id = [
                    tokenizer.eos_token_id,
                    tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>"),
                ]
            # for chatglm3 only
            elif hasattr(tokenizer, "build_prompt"):
                build_prompt = tokenizer.build_prompt(prompt)
                input_ids = tokenizer(
                    [build_prompt] * args.batch_size, return_tensors="pt"
                ).input_ids
            else:
                input_ids = tokenizer(
                    [prompt] * args.batch_size, return_tensors="pt"
                ).input_ids
            gen_ids = user_model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                **generate_kwargs,
                eos_token_id=eos_token_id
            )
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time.time()
            # please check the gen_ids if include input_ids.
            input_tokens_num = input_ids.numel()
            output_tokens_num = torch.tensor(gen_ids).numel() - input_tokens_num
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

    args.model = (
        peft_config.base_model_name_or_path if args.peft_model_id else args.model
    )

    from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import (
        LMEvalParser,
        evaluate,
    )

    args = LMEvalParser(
        model="hf",
        tokenizer=tokenizer,
        user_model=user_model,
        tasks=args.tasks,
        device="cpu",
        batch_size=args.batch_size,
    )
    results = evaluate(args)
    for task_name in args.tasks.split(","):
        if task_name == "wikitext":
            print(
                "Accuracy for %s is: %s"
                % (task_name, results["results"][task_name]["word_perplexity,none"])
            )
        else:
            print(
                "Accuracy for %s is: %s"
                % (task_name, results["results"][task_name]["acc,none"])
            )
