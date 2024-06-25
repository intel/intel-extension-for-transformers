import argparse
import sys
import time
import json
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.utils import check_min_version

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None)
parser.add_argument(
    "--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
)
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
    "--restore",
    action="store_true",
    help="restore ipex quantized model from output_dir/best_configure.json",
)
parser.add_argument(
    "--peft_model_id", type=str, default=None, help="model_name_or_path of peft model"
)
parser.add_argument("--_commit_hash", default=None, type=str)
parser.add_argument("--trust_remote_code", action="store_true")
parser.add_argument("--use_neural_speed", action="store_true")
# ============Benchmark configs==============
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=16, type=int, help="batch size num.")
parser.add_argument(
    "--save_accuracy_path", default=None, help="Save accuracy results path."
)
parser.add_argument("--output_excel", default=None, type=str)
parser.add_argument("--eval_bs", default=4, type=int,
                        help="eval batch size")
parser.add_argument("--tasks", nargs='+', default=["winogrande", "copa", "piqa", "rte", "hellaswag", \
                    "openbookqa", "lambada_openai", "lambada_standard", "wikitext"], type=str, \
                    help="tasks list for accuracy validation")
parser.add_argument("--num_fewshot", default=0, type=int, help="num few shot.")
# ============MixedPrecision configs==============
parser.add_argument("--mixed_precision", action="store_true")

# ============h2o configs==============
parser.add_argument('--h2o', action='store_true')
parser.add_argument('--is_gen', action='store_true')
parser.add_argument('--real_drop', action='store_true')
parser.add_argument("--heavy_ratio", type=float, default=0.1)
parser.add_argument("--recent_ratio", type=float, default=0.1)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--h2o_min_seqlen", type=int, default=0)

args = parser.parse_args()
# transformers version >= 4.32.0 contained the mpt modeling definition.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py
# 4.31.0 for ipex.optimize_transformers
# get model config
if args.peft_model_id:
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(args.peft_model_id)
    if args.model is None:
        args.model = peft_config.base_model_name_or_path
        print("we will use peft base_model_name_or_path to get tokenizer.")

config = AutoConfig.from_pretrained(
    args.model,
    torchscript=False,
    use_cache=True,  # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    _commit_hash=args._commit_hash,
)

# chatglm
if config.model_type == "chatglm":
    AutoModelForCausalLM = AutoModel
# tokenizer
if config.model_type == "llama":
    from transformers import LlamaTokenizer

    # tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

# use peft
args.model = args.peft_model_id if args.peft_model_id is not None else args.model

# Generation
if args.use_neural_speed:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
else:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

if 'cpu' in args.device:
    device = args.device
else:
    device = f"cuda:{args.device}"

# get optimized model
if args.h2o:
    print('Enable Small Cache Size')
    from intel_extension_for_transformers.transformers.kv_cache_compression import H2OConfig, LlamaForCausalLM
    h2o_config = H2OConfig(
        heavy_ratio=args.heavy_ratio,
        recent_ratio=args.recent_ratio,
        h2o_min_seqlen=args.h2o_min_seqlen,
        real_drop=args.real_drop,
        mean=False,
    )
    user_model = LlamaForCausalLM.from_pretrained(
        args.model,
        prune_config=h2o_config,
        trust_remote_code=args.trust_remote_code)
    print("converted model: ", user_model)
else:
    user_model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
user_model.to(device)

# save model
# if args.output_dir is not None:
#     tokenizer.save_pretrained(args.output_dir)
#     user_model.save_pretrained(args.output_dir)

if args.benchmark:
    user_model = (
        user_model.eval() if (not (args.int8 or args.int8_bf16_mixed) and hasattr(user_model, "eval")) else user_model
    )
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
            if hasattr(tokenizer, "build_chat_input"):
                input_ids = tokenizer.build_chat_input(prompt)["input_ids"]
                input_ids = input_ids.repeat(args.batch_size, 1)
                eos_token_id = [
                    tokenizer.eos_token_id,
                    tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>"),
                ]
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
    user_model = (user_model.eval() if (not (args.int8 or args.int8_bf16_mixed) and hasattr(user_model, "eval")) \
                  else user_model)
    # from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
    # model_args="pretrained="+args.model+",trust_remote_code="+str(args.trust_remote_code)
    # args.tasks = ",".join(args.tasks)
    # tokenizer.pad_token = tokenizer.eos_token
    # eval_args = LMEvalParser(model = "hf", 
    #                     user_model=user_model,
    #                     tokenizer=tokenizer,
    #                     model_args=model_args,
    #                     tasks = args.tasks,
    #                     device = device,
    #                     num_fewshot=args.num_fewshot,
    #                     output_path=args.save_accuracy_path,
    #                     batch_size = args.batch_size)
    # print("using device:", device)
    # results = evaluate(eval_args)


    # original lm_eval
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager
    import lm_eval

    verbosity = 'INFO'
    task_manager = TaskManager(verbosity)
    limit = None
    cache_requests = False
    lm = lm_eval.api.registry.get_model("hf")(
                pretrained=user_model,
                batch_size=args.batch_size,
                max_batch_size=None,
            )
    model_args="pretrained="+ args.model+ ",tokenizer="+ args.model + ",dtype=float32"
    use_cache = None
    results = simple_evaluate(
        model=lm,
        model_args=model_args,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        device=device
    )
    import pprint
    pprint.pprint(results["results"])
