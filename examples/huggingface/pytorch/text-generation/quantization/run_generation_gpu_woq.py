import argparse
import re
import time
import json
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.generation import GenerationConfig
import intel_extension_for_pytorch as ipex
from intel_extension_for_transformers.llm.utils.generation import _beam_search, _greedy_search
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
from intel_extension_for_transformers.llm.quantization.utils import convert_dtype_str2torch
from transformers.utils import check_min_version

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="Qwen/Qwen-7B-Chat", const="Qwen/Qwen-7B-Chat"
)
parser.add_argument("--revision", default=None, type=str)
parser.add_argument("--trust_remote_code", default=True)
parser.add_argument(
    "--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--num_beams", default=1, type=int, help="number of beams"
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
parser.add_argument("--do_profiling", action="store_true")
parser.add_argument("--disable_optimize_transformers", action="store_true")
parser.add_argument("--profile_token_latency", action="store_true")
parser.add_argument("--iters", default=10, type=int, help="num iter")
parser.add_argument("--num_warmup", default=3, type=int, help="num warmup")
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="batch size num.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai"], type=str, \
                    help="tasks list for accuracy validation")
# ============WeightOnlyQuant configs===============
parser.add_argument("--woq", action="store_true")
parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'GPTQ'], 
                    help="Weight-only parameter.")
parser.add_argument("--woq_dtype", type=str, default="int4_fullrange",
                    choices=["int4_fullrange"])
parser.add_argument("--woq_group_size", type=int, default=32)
parser.add_argument("--woq_scheme", default="sym")
parser.add_argument("--woq_enable_mse_search", action="store_true")
parser.add_argument("--device", default="xpu")
parser.add_argument("--compute_dtype", default="fp16")
parser.add_argument(
    "--gptq_percdamp",
    type=float,
    default=0.01,
    help="Percent of the average Hessian diagonal to use for dampening.",
)
parser.add_argument(
    "--gptq_block_size",
    type=int,
    default=128,
    help="Block size. sub weight matrix size to run GPTQ.",
)
parser.add_argument(
    "--gptq_nsamples", type=int, default=128, help="Number of calibration data samples."
)
parser.add_argument(
    "--gptq_use_max_length",
    action="store_true",
    help="Set all sequence length to be same length of args.gptq_pad_max_length",
)
parser.add_argument(
    "--gptq_pad_max_length",
    type=int,
    default=2048,
    help="Calibration dataset sequence max length, this should align with your model config",
)
# ============BitsAndBytes configs==============
parser.add_argument("--bitsandbytes", action="store_true")
parser.add_argument("--load_in_4bit", type=bool, default=False)
parser.add_argument("--load_in_8bit", type=bool, default=False)
# =======================================
args = parser.parse_args()
torch_dtype = convert_dtype_str2torch(args.compute_dtype)

# transformers version >= 4.32.0 contained the mpt modeling definition.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py
check_min_version("4.31.0")

# get model config
config = AutoConfig.from_pretrained(
    args.model,
    use_cache=True, # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    revision=args.revision,
)

user_model = None

# tokenizer
if config.model_type == "llama":
   from transformers import LlamaTokenizer
   tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
   tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

quantization_config = None
if args.woq:
    if args.woq_algo == "GPTQ":
        algorithm_args = {
            "act_order": False,
            "percdamp": args.gptq_percdamp,
            "block_size": args.gptq_block_size,
            "nsamples": args.gptq_nsamples,
            "use_max_length": args.gptq_use_max_length,
            "pad_max_length": args.gptq_pad_max_length,
        }
        quantization_config = WeightOnlyQuantConfig(
            compute_dtype=args.compute_dtype,
            scale_dtype=args.compute_dtype,
            weight_dtype=args.woq_dtype,
            scheme=args.woq_scheme,
            group_size=args.woq_group_size,
            algorithm=args.woq_algo,
            tokenizer=tokenizer,
            algorithm_args=algorithm_args,
        )
    else:
        quantization_config = WeightOnlyQuantConfig(
            compute_dtype=args.compute_dtype, weight_dtype=args.woq_dtype,
            group_size=args.woq_group_size, scale_dtype=args.compute_dtype
        ) #default is A16W4G16

# get model
if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                      device_map=args.device,
                                                      quantization_config=quantization_config,
                                                      trust_remote_code=args.trust_remote_code,
                                                      torch_dtype=torch.float16,
                                                      use_neural_speed=False
                                                      )
elif args.load_in_4bit or args.load_in_8bit:
    # CPU device usage is provided by intel-extension-for-transformers.
    user_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                      device_map=args.device,
                                                      load_in_4bit=args.load_in_4bit,
                                                      load_in_8bit=args.load_in_8bit,
                                                      use_neural_speed=False
                                                      )
if user_model is not None:
    user_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if args.benchmark:
    if config.model_type == "qwen":
        prompt = "它完成了，并提交了。你可以在Android和网络上玩美味生存。在网络上玩是有效的，但你必须模拟多次触摸才能移动桌子."
    else:
        prompt = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    user_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, device_map=args.device, torch_dtype=torch_dtype) \
            if user_model is None else user_model
    user_model = user_model.to(memory_format=torch.channels_last)
    if not args.disable_optimize_transformers:
        print("Optimize with IPEX...")
        user_model = ipex.optimize_transformers(
            user_model.eval(), device=args.device, inplace=True, woq=(hasattr(user_model, "quantization_config")), dtype=torch_dtype)
    else:
        print("Disabled optimization with IPEX...")
    # start
    num_iter = args.iters
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    amp_enabled = True
    amp_dtype = torch_dtype

    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=args.num_beams)
    if args.profile_token_latency:
        ipex.transformers.optimize.convert_function(user_model, "greedy_search", _greedy_search)
        if args.disable_optimize_transformers:
            ipex.transformers.optimize.convert_function(user_model, "beam_search", _beam_search)
        user_model.config.token_latency = True

    total_time = 0.0
    total_list = []
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        for i in range(num_iter + num_warmup):
            with torch.autograd.profiler_legacy.profile(enabled=args.do_profiling, use_xpu=(args.device=="xpu"), record_shapes=False) as prof:
                input_ids = tokenizer(
                    prompt, return_tensors="pt").input_ids.to(args.device)
                tic = time.time()
                output = user_model.generate(
                    input_ids, max_new_tokens=int(args.max_new_tokens), **generate_kwargs
                )
                toc = time.time()
                gen_ids = output[0] if args.profile_token_latency else output
                gen_text = tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True)
                if args.device == "xpu":
                    torch.xpu.synchronize()
            if args.do_profiling and i >= num_warmup and (i == num_warmup or i == num_iter + num_warmup - 1):
                print(f"Save pt for iter {i}")
                torch.save(prof.key_averages().table(
                    sort_by="self_xpu_time_total"), f"./profile_{i}.pt")
                # torch.save(prof.table(sort_by="id", row_limit=-1),
                #            './profile_id.pt')
                # torch.save(prof.key_averages(
                #     group_by_input_shape=True).table(), "./profile_detail.pt")
                prof.export_chrome_trace(f"./trace_{i}.json")
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o - i if user_model.config.model_type != "t5" else o
                for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            print(gen_text, total_new_tokens, flush=True)
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
            if i >= num_warmup:
                total_time += toc - tic
                if args.profile_token_latency:
                    total_list.append(output[1])

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter - num_warmup)
    print("Inference latency: %.5f sec." % latency)
    throughput = (args.max_new_tokens + input_size) / latency
    print("Average throughput: {} samples/sec".format(throughput))

    if args.profile_token_latency:
        import numpy as np
        from itertools import chain

        first_latency = np.mean([x[0] for x in total_list])
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n)
        print("First token average latency: %.5f sec." % first_latency)
        print("Average 2... latency: %.5f sec." % average_2n_latency)
        print(total_list)


if args.accuracy:
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, device_map=args.device, torch_dtype=torch_dtype) \
            if user_model is None else user_model
    if not args.disable_optimize_transformers:
        print("Optimize with IPEX...")
        user_model = ipex.optimize_transformers(
            user_model.eval(), device=args.device, inplace=True, woq=(hasattr(user_model, "quantization_config")), dtype=torch_dtype)
    else:
        print("Disabled optimization with IPEX...")
    results = evaluate(
        model="hf-causal",
        model_args='pretrained='+args.model+',tokenizer=' + args.model + \
            ',dtype=float32,trust_remote_code=' + str(args.trust_remote_code),
        user_model=user_model,
        batch_size=args.batch_size,
        tasks=args.tasks,
        device=args.device
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

