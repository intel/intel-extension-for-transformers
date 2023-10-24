import argparse
import re
import time
import json
import os
import pathlib
import torch
import types
from pathlib import Path
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PretrainedConfig
import transformers
from optimum.utils import NormalizedConfigManager

import numpy as np
from itertools import chain

parser = argparse.ArgumentParser()

# Main config
parser.add_argument(
    "--model", nargs="?", default="bigcode/starcoderbase", const="bigcode/starcoderbase"
)
parser.add_argument(
    "--dataset", nargs="?", default="mbpp", const="mbpp"
)
parser.add_argument(
    "--calib_split", nargs="?", default="test", const="test"
)
parser.add_argument("--dtype", type=str, default="int8")
parser.add_argument(
    "--max_new_tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default="auto", help="Smooth quant parameter.")
parser.add_argument(
    "--pad_max_length", default=512, type=int, help="Pad input ids to max length."
)
parser.add_argument("--calib_iters", default=32, type=int, help="calibration iters.")
parser.add_argument("--calib_batch_size", default=1, type=int, help="calibration iters.")
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
parser.add_argument("--save_generations_path", default=None)
parser.add_argument("--load_generations_path", default=None)
parser.add_argument("--metric_output_path", default="evaluation_results.json")
parser.add_argument("--seed", default=0, type=int)


# Generation config
parser.add_argument("--max_length_generation", default=512, type=int)
parser.add_argument("--temperature", default=0.8, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
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

# quantize
if args.quantize:
    def generate_dummy_past_key_values(input_bs, user_model):
        normalized_config = NormalizedConfigManager.get_normalized_config_class(
            user_model.config.model_type
        )(user_model.config)
        nb_pkv = 2
        num_layers = normalized_config.num_layers
        num_attention_heads = normalized_config.num_attention_heads
        hidden_size = normalized_config.hidden_size
        d_k = hidden_size // num_attention_heads

        if user_model.config.model_type == "gpt_bigcode":
            new_shape = [input_bs, 1, d_k*2]
            dummy_tensor = torch.ones(size=new_shape)
            past_key_values = tuple([dummy_tensor] * num_layers)

        elif user_model.config.model_type != "bloom":
            new_shape = [input_bs, num_attention_heads, 1, d_k]
            dummy_tensor = torch.ones(size=new_shape)
            past_key_values = tuple(
                tuple(dummy_tensor for _ in range(nb_pkv)) for _ in range(num_layers)
            )
            pkv = tuple(dummy_tensor for _ in range(nb_pkv))
            past_key_values = tuple(tuple(pkv) for _ in range(num_layers))
        else:
            pkv = ()
            for nb_pkv in range(nb_pkv):
                if nb_pkv % 2 == 0:
                    new_shape = [input_bs * num_attention_heads, d_k, 1]
                else:
                    new_shape = [input_bs * num_attention_heads, 1, d_k]
                pkv = pkv + (torch.ones(size=new_shape),)
            past_key_values = tuple(tuple(pkv) for _ in range(num_layers))
        return past_key_values

    class Evaluator:
        def __init__(
            self,
            dataset,
            tokenizer,
            batch_size=8,
            pad_val=1,
            pad_max=512,
            is_calib=False,
        ):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.pad_val = pad_val
            self.pad_max = pad_max
            self.is_calib = is_calib

            # tokenize the dataset
            self.dataset = self.dataset.map(self.tokenize_function, batched=True)
            self.dataset.set_format(type="torch", columns=["input_ids"])

        @torch.no_grad()
        def tokenize_function(self, examples):
            if 'prompt' in examples:
                example = self.tokenizer(examples["prompt"])
            elif 'text' in examples:
                example = self.tokenizer(examples["text"])
            elif 'code' in examples:
                example = self.tokenizer(examples["code"])
            else:
                print("Check dataset prompt identifier")
            return example

        @torch.no_grad()
        def collate_batch(self, batch):
            input_ids_padded = []
            last_ind = []
            for text in batch:
                input_ids = text["input_ids"]
                pad_len = self.pad_max - input_ids.shape[0]
                last_ind.append(input_ids.shape[0] - 1)
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
                input_ids_padded.append(input_ids)
            return (
                torch.vstack(input_ids_padded),
                torch.tensor(last_ind),
            )

    calib_dataset = load_dataset(args.dataset, split=args.calib_split)
    calib_dataset = calib_dataset.shuffle(seed=42)
    calib_evaluator = Evaluator(
        calib_dataset,
        tokenizer,
        args.batch_size,
        pad_max=args.pad_max_length,
        is_calib=True,
    )
    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=args.calib_batch_size,
        shuffle=False,
        collate_fn=calib_evaluator.collate_batch,
    )
    input_ids = user_model.dummy_inputs["input_ids"]
    input_bs, input_len = input_ids.shape
    past_key_values = generate_dummy_past_key_values(input_bs, user_model)
    attention_mask = torch.ones(input_bs, input_len + 1)
    attention_mask[:,0] = 0
    example_inputs = (
        input_ids,
        tuple(past_key_values),
        attention_mask,
    )

    def calib_func(prepared_model):
        for i, (input_ids, last_ind) in enumerate(calib_dataloader):
            input_bs, input_len = input_ids.shape
            past_key_values = generate_dummy_past_key_values(input_bs, user_model)
            attention_mask = torch.ones(input_bs, input_len + 1)
            attention_mask[:,0] = 0
            if i >= args.calib_iters:
                break
            prepared_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

    from neural_compressor import PostTrainingQuantConfig, quantization


    op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        }

    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    if args.sq:
        args.alpha = args.alpha if args.alpha == "auto" else float(args.alpha)
        recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": args.alpha}}
        conf = PostTrainingQuantConfig(
            backend="ipex" if args.ipex else "default",
            excluded_precisions=excluded_precisions,
            op_type_dict=op_type_dict,
            recipes=recipes,
            example_inputs=example_inputs,
        )
    else:
        conf = PostTrainingQuantConfig(
            backend="ipex" if args.ipex else "default",
            excluded_precisions=excluded_precisions,
            op_type_dict=op_type_dict,
            example_inputs=example_inputs,
        )
    # save config
    user_model.config.save_pretrained(args.output_dir)
    q_model = quantization.fit(
        user_model,
        conf,
        calib_func=calib_func,
    )
    q_model.save(args.output_dir)

# Generation

if args.int8 or args.int8_bf16_mixed:
    # TorchScript model don't attribute generate method, the wrapper is provided.
    if args.ipex:
        user_model = TSModelForCausalLM.from_pretrained(
            args.output_dir, file_name="best_model.pt"
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
