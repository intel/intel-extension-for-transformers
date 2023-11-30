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


def get_example_inputs(tokenized_dataset, model_config, model_type="llama", num_beams=4, ipex_opt_llm=True):
    from intel_extension_for_transformers.transformers.utils.utility import (
    logger,
    LazyImport,
    generate_dummy_past_key_values,
    generate_dummy_past_key_values_for_opt_llm,
    get_example_inputs_for_chatglm
)
    if ipex_opt_llm:
        past_key_values = generate_dummy_past_key_values_for_opt_llm(
                                                                    config=model_config,
                                                                    input_bs=1,
                                                                    num_beams=num_beams
                                                                    )
    else:
        past_key_values = generate_dummy_past_key_values(config=model_config, input_bs=1)
    
    def collate_batch(batch):
        position_ids_padded = []
        input_ids_padded = []
        last_ind = []
        attention_mask_padded = []
        for input_ids in batch:
            input_ids = (
                input_ids[: 512]
                if len(input_ids) > 512
                else input_ids
            )
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
            position_ids_padded.append(position_ids)
            break
        return (
            (
                torch.vstack(input_ids_padded),
                torch.vstack(attention_mask_padded),
                torch.vstack(position_ids_padded),
                past_key_values,
            ),
            torch.tensor(last_ind),
        )
    from torch.utils.data import DataLoader
    calib_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch,
    )
    from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS
    for i, (
            (input_ids, attention_mask, position_ids, past_key_values),
            last_ind,
        ) in enumerate(calib_dataloader):
            if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
                example_inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "position_ids": position_ids,
                            "past_key_values": past_key_values
                        }
            else:
                example_inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "past_key_values": past_key_values
                        }
            break
    return example_inputs

args.accuracy = True
if args.accuracy:
    args.model = peft_config.base_model_name_or_path if args.peft_model_id else args.model

    prompt = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."
    prompt = [prompt] * args.batch_size
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
   
    user_model = AutoModelForCausalLM.from_pretrained(args.model, use_llm_runtime=False, torchscript=True)

    import intel_extension_for_pytorch as ipex
    qconfig = ipex.quantization.default_static_qconfig_mapping
    user_model = ipex.optimize_transformers(
        user_model.eval(),
        dtype=torch.float,
        inplace=True,
        quantization_config=qconfig,
        deployment_mode=False,
    )
    example_inputs = get_example_inputs(input_ids, user_model.config, model_type="llama", ipex_opt_llm=True) 

    from neural_compressor.utils.pytorch import recover_model_from_json
    user_model = recover_model_from_json(user_model, os.path.join(args.output_dir, "best_configure.json"), example_inputs)
    from intel_extension_for_transformers.llm.evaluation.models import TSModelCausalLMForOPTLLM
    config = AutoConfig.from_pretrained(args.output_dir)
    user_model = TSModelCausalLMForOPTLLM(user_model, config=config)

    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    results = evaluate(
        model="hf-causal",
        model_args='pretrained=' + args.model + ',tokenizer=' + args.model + \
            ',dtype=float32' + ",trust_remote_code=" + str(args.trust_remote_code),
        user_model=user_model,
        batch_size= args.batch_size,
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