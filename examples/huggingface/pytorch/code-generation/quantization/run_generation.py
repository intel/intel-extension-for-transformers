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
from transformers import AutoTokenizer, AutoConfig
from optimum.utils import NormalizedConfigManager
from intel_extension_for_transformers.transformers import (
    MixedPrecisionConfig,
    SmoothQuantConfig,
    BitsAndBytesConfig,
    RtnConfig,
    AwqConfig,
    TeqConfig,
    GPTQConfig,
    AutoRoundConfig,
)
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
)
from intel_extension_for_transformers.transformers.utils import str2bool

parser = argparse.ArgumentParser()

# ============Main configs============
parser.add_argument(
    "--model", nargs="?", default="bigcode/starcoderbase", const="bigcode/starcoderbase"
)
parser.add_argument("--trust_remote_code", action="store_true")
parser.add_argument("--_commit_hash", default=None, type=str)
parser.add_argument("--use_neural_speed", action="store_true")
parser.add_argument("--dataset", nargs="?", default="mbpp", const="mbpp")
parser.add_argument("--dtype", type=str, default="int8")
parser.add_argument(
    "--max_new_tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--calib_iters", default=500, type=int, help="calibration iters.")
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
# ============Benchmark configs==============
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
parser.add_argument(
    "--prompt_size", default=32, type=int, help="generate dummy input_ids size"
)
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=56, type=int, help="batch size num.")
parser.add_argument(
    "--save_accuracy_path", default=None, help="Save accuracy results path."
)
# ============MixedPrecision configs==============
parser.add_argument("--mixed_precision", action="store_true")
# ============SmoothQuant configs==============
parser.add_argument("--sq", action="store_true")
parser.add_argument(
    "--calib_padding", action="store_true", help="Calibration dataset do padding."
)
parser.add_argument(
    "--calib_shuffle",
    default=True,
    type=str2bool,
    help="Calibration dataset do shuffle.",
)
parser.add_argument(
    "--calib_pad_val", default=1, type=int, help="Calibration dataset padding value."
)
parser.add_argument(
    "--calib_len",
    default=512,
    type=int,
    help="Calibration dataset max or padding max length.",
)
parser.add_argument(
    "--recipes", type=str, help="A dictionary as a string, recipes for smoothquant."
)
parser.add_argument("--alpha", default="0.5", help="Smooth quant parameter.")
# ============BitsAndBytes configs==============
parser.add_argument("--bitsandbytes", action="store_true")
# ============WeightOnlyQuant configs===============
parser.add_argument("--woq", action="store_true")
parser.add_argument(
    "--woq_algo",
    default="Rtn",
    choices=["Rtn", "Awq", "Teq", "GPTQ", "AutoRound"],
    help="Weight-only algorithm.",
)
parser.add_argument(
    "--bits",
    type=int,
    default=8,
    choices=[4, 8],
)
parser.add_argument(
    "--weight_dtype",
    type=str,
    default="int8",
    choices=[
        "int8",
        "int4_clip",
        "int4_fullrange",
        "fp4_e2m1_bnb",
        "fp4_e2m1",
        "nf4",
        "fp8_e5m2",
        "fp8_e4m3",
    ],
)
parser.add_argument(
    "--scale_dtype",
    type=str,
    default="fp32",
    choices=["fp32", "fp8"],
)
parser.add_argument(
    "--compute_dtype",
    type=str,
    default="fp32",
    choices=["fp32", "bf16", "int8"],
)
parser.add_argument("--group_size", type=int, default=32)
parser.add_argument("--scheme", default="sym")
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument(
    "--layer_wise",
    action="store_true",
    help="Use layer wise to do quantization",
)
# ============GPTQ configs==============
parser.add_argument(
    "--desc_act",
    action="store_true",
    help="Whether to apply the activation order GPTQ heuristic.",
)
parser.add_argument(
    "--damp_percent",
    type=float,
    default=0.01,
    help="Percent of the average Hessian diagonal to use for dampening.",
)
parser.add_argument(
    "--blocksize",
    type=int,
    default=128,
    help="Block size. sub weight matrix size to run GPTQ.",
)
parser.add_argument(
    "--nsamples", type=int, default=128, help="Number of calibration data samples."
)
parser.add_argument(
    "--max_input_length",
    type=int,
    default=2048,
    help="Calibration dataset sequence max length, this should align with your model config",
)
parser.add_argument(
    "--static_groups",
    action="store_true",
    help="Use determined group to do quantization",
)
# ============AUTOROUND configs==============
parser.add_argument(
    "--lr",
    type=float,
    default=0.0025,
    help="learning rate, if None, it will be set to 1.0/iters automatically",
)
parser.add_argument(
    "--minmax_lr",
    type=float,
    default=0.0025,
    help="minmax learning rate, if None,it will beset to be the same with lr",
)
parser.add_argument(
    "--use_quant_input",
    action="store_true",
    help="whether to use the output of quantized block to tune the next block",
)
# ============Harness configs============
parser.add_argument("--tasks", default=None, help="Evaluation tasks")
parser.add_argument(
    "--limit", default=None, type=int, help="Limit number of samples to eval"
)
parser.add_argument("--allow_code_execution", action="store_true")
parser.add_argument("--generation_only", action="store_true")
parser.add_argument("--postprocess", action="store_false")
parser.add_argument("--save_references", action="store_true")
parser.add_argument("--save_generations", action="store_true")
parser.add_argument("--instruction_tokens", default=None)
parser.add_argument("--save_generations_path", default="generations.json")
parser.add_argument("--load_generations_path", default=None)
parser.add_argument("--metric_output_path", default="evaluation_results.json")
parser.add_argument(
    "--load_generations_intermediate_paths",
    type=str,
    nargs="*",
    help="List of paths for saving the intermediate code generations",
)
# ============Generation config============
parser.add_argument("--max_length_generation", default=512, type=int)
parser.add_argument("--check_references", action="store_true")
parser.add_argument("--max_memory_per_gpu", type=str, default=None)
parser.add_argument(
    "--prompt",
    type=str,
    default="prompt",
    help="Prompt type to use for generation in HumanEvalPack tasks",
)
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
parser.add_argument(
    "--save_every_k_tasks",
    type=int,
    default=-1,
    help="Optional saving after every k tasks",
)
parser.add_argument(
    "--left_padding",
    action="store_true",
    help="Force left padding, needed for models like chatglm3-6b",
)
parser.add_argument(
    "--load_data_path",
    type=str,
    default=None,
    help="Path of additional data to load for the tasks",
)
# ============Evaluation configs==============
parser.add_argument("--prefix", default="")
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--temperature", default=0.2, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--top_k", default=0, type=int)
parser.add_argument("--n_samples", default=1, type=int)
parser.add_argument("--eos", default="<|endoftext|>", type=str)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    truncation_side="left",
    padding_side="right",
    trust_remote_code=args.trust_remote_code,
    _commit_hash=args._commit_hash,
)

config = AutoConfig.from_pretrained(
    args.model,
    torchscript=(
        True
        if (
            args.sq
            or args.woq_algo in ["Awq", "Teq"]
            or (args.int8 or args.int8_bf16_mixed or args.benchmark)
        )
        else False
    ),  # torchscript will force `return_dict=False` to avoid jit errors
    use_cache=True,  # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    _commit_hash=args._commit_hash,
)
if not tokenizer.eos_token:
    if tokenizer.bos_token:
        tokenizer.eos_token = tokenizer.bos_token
        print("bos_token used as eos_token")
    else:
        raise ValueError("No eos_token or bos_token found")

tokenizer.pad_token = tokenizer.eos_token


op_type_dict = {
    "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
}
recipes = {
    "smooth_quant": True,
    "smooth_quant_args": {
        "alpha": args.alpha if args.alpha == "auto" else float(args.alpha)
    },
}
excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
# mp/sq/woq/bitsandbytes config setting
quantization_config = None
if args.mixed_precision:
    quantization_config = MixedPrecisionConfig(dtype="bfloat16")  # default is bfloat16
elif args.sq:
    quantization_config = SmoothQuantConfig(
        tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
        recipes=recipes,
        op_type_dict=op_type_dict,  # default is {}
        excluded_precisions=excluded_precisions,  # default is []
        calib_dataset=args.dataset,
        calib_iters=args.calib_iters,
    )
elif args.woq:
    if args.woq_algo == "Rtn":
        quantization_config = RtnConfig(
            tokenizer=tokenizer,
            bits=args.bits,
            sym=True if args.scheme == "sym" else False,
            group_size=args.group_size,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            layer_wise=args.layer_wise,
        )
    elif args.woq_algo == "Awq":
        quantization_config = AwqConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            zero_point=False if args.scheme == "sym" else True,
            group_size=args.group_size,
            max_input_length=args.max_input_length,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            calib_iters=args.calib_iters,
        )
    elif args.woq_algo == "Teq":
        quantization_config = TeqConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            sym=True if args.scheme == "sym" else False,
            group_size=args.group_size,
            max_input_length=args.max_input_length,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            calib_iters=args.calib_iters,
        )
    elif args.woq_algo == "GPTQ":
        quantization_config = GPTQConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            desc_act=args.desc_act,
            damp_percent=args.damp_percent,
            sym=True if args.scheme == "sym" else False,
            blocksize=args.blocksize,
            nsamples=args.nsamples,
            static_groups=args.static_groups,
            group_size=args.group_size,
            max_input_length=args.max_input_length,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            calib_iters=args.calib_iters,
            layer_wise=args.layer_wise,
        )
    elif args.woq_algo == "AutoRound":
        quantization_config = AutoRoundConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            sym=True if args.scheme == "sym" else False,
            nsamples=args.nsamples,
            group_size=args.group_size,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            calib_iters=args.calib_iters,
            calib_len=args.calib_len,
            lr=args.lr,
            minmax_lr=args.minmax_lr,
            use_quant_input=args.use_quant_input,
        )
    else:
        assert False, "Please set the correct '--woq_algo'"
# bitsandbytes
elif args.bitsandbytes:
    # GPU device is need for `load_in_4bit` and `load_in_8bit`.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

# get optimized model
if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
        use_neural_speed=False,
    )
elif args.load_in_4bit or args.load_in_8bit:
    # CPU device usage is provided by intel-extension-for-transformers.
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        _commit_hash=args._commit_hash,
        use_neural_speed=False,
    )
elif not args.int8 and not args.int8_bf16_mixed:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
        use_neural_speed=False,
    )

# save model
if args.output_dir is not None:
    tokenizer.save_pretrained(args.output_dir)
    if args.sq:
        config.save_pretrained(args.output_dir)
        user_model.save(args.output_dir)
    elif args.mixed_precision or args.woq:
        # user_model will be changed.
        user_model.save_pretrained(args.output_dir)
        # loading saved woq model
        user_model = AutoModelForCausalLM.from_pretrained(
            args.output_dir, 
            trust_remote_code=args.trust_remote_code,
            use_neural_speed=args.use_neural_speed
            )

if args.int8 or args.int8_bf16_mixed:
    # TorchScript model don't attribute generate method, the wrapper is provided.
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_transformers.transformers.llm.evaluation.models import (
        TSModelCausalLMForITREX,
    )

    user_model = TSModelCausalLMForITREX.from_pretrained(
        args.output_dir,
        file_name="best_model.pt",
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
    )

if args.benchmark:
    normalized_config = NormalizedConfigManager.get_normalized_config_class(
        user_model.config.model_type
    )(user_model.config)
    num_layers = normalized_config.num_layers
    num_attention_heads = normalized_config.num_attention_heads
    hidden_size = normalized_config.hidden_size
    d_k = hidden_size // num_attention_heads
    num_beams = 1
    if hasattr(normalized_config, "num_key_value_heads"):
        num_key_value_heads = normalized_config.num_key_value_heads
    if hasattr(normalized_config, "multi_query_group_num"):
        num_key_value_heads = normalized_config.multi_query_group_num

    num_iter = args.iters
    num_warmup = args.num_warmup

    total_latency = 0
    for j in range(args.max_new_tokens):
        total_time = 0.0
        with torch.inference_mode(), torch.no_grad():
            for i in range(num_iter):
                tic = time.time()
                if j == 0:
                    input_ids = torch.randint(
                        1,
                        tokenizer.vocab_size,
                        size=(args.batch_size, args.prompt_size),
                    )
                    input_bs, input_len = input_ids.shape
                    attention_mask = torch.ones(input_bs, input_len)
                    position_ids = (
                        torch.arange(input_len).unsqueeze(0).expand(input_bs, -1)
                    )
                    if user_model.config.model_type == "gpt_bigcode":
                        new_shape = [input_bs, 0, d_k * 2]
                        dummy_tensor = torch.zeros(size=new_shape)
                        past_key_values = tuple([dummy_tensor] * num_layers)
                    else:
                        if not (args.int8 or args.int8_bf16_mixed):
                            new_shape = [input_bs, num_key_value_heads, 0, d_k]
                            past_key_values = [
                                (
                                    torch.zeros(size=new_shape).contiguous(),
                                    torch.zeros(size=new_shape).contiguous(),
                                )
                                for _ in range(num_layers)
                            ]
                            past_key_values = tuple(past_key_values)

                        else:
                            new_shape = [input_bs, num_key_value_heads, 1, d_k]
                            beam_idx_tmp = torch.zeros(
                                (2048, int(input_bs * num_beams)), dtype=torch.long
                            ).contiguous()
                            past_key_values = [
                                (
                                    torch.zeros(
                                        1, 0, 0, 1, dtype=torch.long
                                    ).contiguous(),
                                    torch.zeros(size=new_shape).contiguous(),
                                    torch.zeros(size=new_shape).contiguous(),
                                    beam_idx_tmp,
                                )
                                for _ in range(num_layers)
                            ]
                            past_key_values = tuple(past_key_values)

                inp = {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
                out = user_model(**inp)
                gen_id = torch.argmax(out[0][:, -1:, :], axis=-1)
                gen_text = tokenizer.batch_decode(gen_id, skip_special_tokens=True)
                toc = time.time()
                if i >= num_warmup:
                    total_time += toc - tic

        print("\n", "-" * 10, "Summary:", "-" * 10)
        print("Generated token index:", j + 1)
        latency = total_time / (num_iter - num_warmup)
        print("Inference latency: %.5f sec." % latency)
        throughput = (num_iter - num_warmup) / total_time
        print("Throughput: {} samples/sec".format(throughput))

        input_ids = gen_id
        past_key_values = out[1]
        attention_mask = torch.ones(
            (attention_mask.shape[0], attention_mask.shape[1] + 1)
        )
        position_ids = torch.tensor([[len(inp["position_ids"])]])
        total_latency += latency

    average_latency = total_latency / args.max_new_tokens
    print("Average inference latency: %.5f sec." % latency)
    average_throughput = args.max_new_tokens / total_latency
    print("Average throughput: {} samples/sec".format(throughput))


if args.accuracy:
    from intel_extension_for_transformers.transformers.llm.evaluation.bigcode_eval import evaluate

    results = evaluate(
        model=user_model,
        tokenizer=tokenizer,
        tasks=args.tasks,
        batch_size=args.batch_size,
        args=args,
    )
    print(results)
