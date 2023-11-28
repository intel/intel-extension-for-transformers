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
from optimum.intel.generation.modeling import TSModelForCausalLM
from intel_extension_for_transformers.transformers import (
    MixedPrecisionConfig,
    WeightOnlyQuantConfig,
    SmoothQuantConfig,
)
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
    AutoModel
)

parser = argparse.ArgumentParser()

# ============Main configs============
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
parser.add_argument("--calib_iters", default=32, type=int, help="calibration iters.")
parser.add_argument("--calib_batch_size", default=1, type=int, help="calibration batch size.")
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--quantized_model_path", default="./saved_results/best_model.pt",
                    help="path to quantized pt file")
# ============Benchmark configs==============
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
parser.add_argument("--prompt_size", default=32, type=int, help="generate dummy input_ids size")
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=56, type=int,
                    help="batch size num.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
# ============MixedPrecision configs==============
parser.add_argument("--mixed_precision", action="store_true")
# ============SmoothQuant configs==============
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default="0.5", help="Smooth quant parameter.")
# ============WeightOnlyQuant configs============
parser.add_argument("--woq", action="store_true")
parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ'], 
                    help="Weight-only parameter.")
parser.add_argument("--woq_dtype", type=str, default="int4_fullrange", 
                    choices=["int8", "int4_clip", "int4_fullrange", "fp4_e2m1_bnb", "fp4_e2m1", "nf4"])
parser.add_argument("--woq_group_size", type=int, default=32)
parser.add_argument("--woq_scheme", default="sym")
# ============Harness configs============
parser.add_argument("--tasks", default=None, help="Evaluation tasks", choices=["mbpp", "humaneval"])
parser.add_argument("--n_samples", default=200, type=int)
parser.add_argument("--limit", default=None, type=int, help="Limit number of samples to eval")
parser.add_argument("--allow_code_execution", action="store_true")
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
# ============Generation config============
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


tokenizer = AutoTokenizer.from_pretrained(
    args.model,
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


calib_dataset = args.dataset
op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
                }
recipes = {
            "smooth_quant": True,
            "smooth_quant_args": {"alpha": args.alpha if args.alpha == "auto" else float(args.alpha)},
            }
excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
# mp/sq/woq/bitsandbytes config setting
quantization_config = None
if args.mixed_precision:
    quantization_config = MixedPrecisionConfig(dtype="bfloat16") # default is bfloat16
elif args.sq:
    quantization_config = SmoothQuantConfig(
        tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
        recipes=recipes,
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
    elif args.mixed_precision:
        user_model.config.save_pretrained(args.output_dir)
        torch.save(user_model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))


if args.int8 or args.int8_bf16_mixed:
    # TorchScript model don't attribute generate method, the wrapper is provided.
    import intel_extension_for_pytorch as ipex
    if config.model_type == "llama":
        if args.accuracy:
            from intel_extension_for_transformers.llm.evaluation.models import TSModelCausalLMForOPTLLM
            user_model = TSModelCausalLMForOPTLLM.from_pretrained(
                args.output_dir, file_name="best_model.pt", trust_remote_code=args.trust_remote_code
            )
        else:
            torch._C._jit_set_texpr_fuser_enabled(False)
            qconfig = ipex.quantization.default_static_qconfig_mapping
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
                if j==0:
                    input_ids = torch.randint(1, tokenizer.vocab_size, size = (args.batch_size, args.prompt_size))
                    input_bs, input_len = input_ids.shape
                    attention_mask = torch.ones(input_bs, input_len)
                    position_ids = torch.arange(input_len).unsqueeze(0).expand(input_bs, -1)
                    if user_model.config.model_type == "gpt_bigcode":
                        new_shape = [input_bs, 0, d_k*2]
                        dummy_tensor = torch.zeros(size=new_shape)
                        past_key_values = tuple([dummy_tensor] * num_layers)
                    else:
                        new_shape = [input_bs, num_key_value_heads, 1, d_k]
                        beam_idx_tmp = torch.zeros(
                                    (2048, int(input_bs * num_beams)), dtype=torch.long
                                ).contiguous()
                        past_key_values = [(torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros(size=new_shape).contiguous(),
                                torch.zeros(size=new_shape).contiguous(),
                                beam_idx_tmp) for _ in range(num_layers)]
                        past_key_values = tuple(past_key_values)
              
                inp = {"input_ids": input_ids,
                        "past_key_values": past_key_values,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids}
                out = user_model(**inp)
                gen_id = torch.argmax(out[0][:, -1:, :], axis = -1)
                gen_text = tokenizer.batch_decode(gen_id, skip_special_tokens=True)
                toc = time.time()
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
        position_ids = torch.tensor([[len(inp["position_ids"])]])
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
