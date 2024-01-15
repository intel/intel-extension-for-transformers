import argparse
import re
import time
import json
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.generation import GenerationConfig
import intel_extension_for_pytorch as ipex
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from transformers.utils import check_min_version
from intel_extension_for_transformers.transformers import (
    WeightOnlyQuantConfig,
)

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
parser.add_argument("--num_warmup", default=0, type=int, help="num warmup")
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=56, type=int,
                    help="batch size num.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai"], type=str, \
                    help="tasks list for accuracy validation")
# ============WeightOnlyQuant configs===============
parser.add_argument("--woq", action="store_true")
parser.add_argument("--woq_algo", default="RTN", choices=['RTN'], 
                    help="Weight-only parameter.")
parser.add_argument("--woq_dtype", type=str, default="int4_fullrange",
                    choices=["int4_fullrange"])
parser.add_argument("--woq_group_size", type=int, default=64)
parser.add_argument("--woq_scheme", default="sym")
parser.add_argument("--woq_enable_mse_search", action="store_true")
parser.add_argument("--device", default="cpu")
parser.add_argument("--compute_dtype", default="fp16")
# ============BitsAndBytes configs==============
parser.add_argument("--bitsandbytes", action="store_true")
parser.add_argument("--load_in_4bit", type=bool, default=False)
parser.add_argument("--load_in_8bit", type=bool, default=False)
# =======================================
args = parser.parse_args()
torch_dtype = torch.float16 if args.compute_dtype == "fp16" else torch.float32

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
generation_config = GenerationConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
generation_config.do_sample = False
user_model = None

# tokenizer
if config.model_type == "llama":
   from transformers import LlamaTokenizer
   tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
   tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

quantization_config = None
if args.woq:
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
                                                      fp16=True,
                                                      use_llm_runtime=False
                                                      )
elif args.load_in_4bit or args.load_in_8bit:
    # CPU device usage is provided by intel-extension-for-transformers.
    user_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                      device_map=args.device,
                                                      load_in_4bit=args.load_in_4bit,
                                                      load_in_8bit=args.load_in_8bit,
                                                      use_llm_runtime=False
                                                      )
tokenizer.save_pretrained(args.output_dir)
if user_model is not None:
    user_model.save_low_bit(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if args.benchmark:
    prompt = "也许你能给我介绍一下中国的首都么？"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    user_model = AutoModelForCausalLM.load_low_bit(args.model, trust_remote_code=True) if user_model is None else user_model
    user_model = ipex.optimize_transformers(user_model.eval(), device=args.device)
    # start
    total_time = 0.0
    num_iter = args.iters
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    total_token_num = 0

    total_latency = 0
    first_token_latency = 0
    gen_texts = []
    for j in range(args.max_new_tokens):
        total_time = 0.0
        with torch.inference_mode(), torch.no_grad():
            for i in range(num_iter):
                if j==0:
                    inp = tokenizer(prompt, return_tensors="pt").to(args.device)
                    attention_mask = inp["attention_mask"] if "attention_mask" in inp else torch.ones(inp["input_ids"].shape)
                else:
                    inp = {"input_ids": input_ids,
                            "past_key_values": past_key_values,
                            "attention_mask": attention_mask}
                tic = time.time()
                out = user_model(**inp)
                toc = time.time()
                gen_id = torch.argmax(out[0][:, -1:, :], axis = -1).to("cpu")
                gen_text = tokenizer.batch_decode(gen_id, skip_special_tokens=True)
                if i >= num_warmup:
                    total_time += toc - tic
            gen_texts.extend(gen_text)
        latency = total_time / (num_iter - num_warmup) / args.batch_size
        throughput = (num_iter - num_warmup) / total_time
        if j == 0:
            print("\n", "-" * 10, "Summary:", "-" * 10)
            print("Generated token index:", j+1)
            print("Inference latency: %.5f sec." % latency)
            print("Throughput: {} samples/sec".format(throughput))
            first_token_latency = latency
        input_ids = gen_id.to(args.device)
        past_key_values = out[1]
        attention_mask = torch.ones((attention_mask.shape[0], attention_mask.shape[1] + 1)).to(args.device)
        total_latency += latency

    print(prompt[0] + ":\n" + "".join(gen_texts))
    print("first token inference latency: %.5f sec." % first_token_latency)
    next_token_latency = (total_latency - first_token_latency) / (args.max_new_tokens - 1)
    print("next token inference latency: %.5f sec." % next_token_latency)
    average_latency = total_latency / args.max_new_tokens
    print("Average inference latency: %.5f sec." % latency)
    average_throughput = args.max_new_tokens / total_latency
    print("Average throughput: {} samples/sec".format(throughput))

if args.accuracy:
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    user_model = AutoModelForCausalLM.load_low_bit(args.model, trust_remote_code=True) if user_model is None else user_model
    user_model = ipex.optimize_transformers(user_model.eval(), device=args.device)
    results = evaluate(
        model="hf-causal",
        model_args='pretrained='+args.model+',tokenizer='+args.model+',dtype=float32',
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

