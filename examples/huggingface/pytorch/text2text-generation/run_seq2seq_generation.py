#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import copy
import logging
import json
import numpy as np
import re
import time
import torch
from datasets import load_dataset, load_from_disk
from optimum.exporters import TasksManager
from optimum.intel.neural_compressor import INCConfig
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from intel_extension_for_transformers.optimization.modeling import INCModelForSeq2SeqLM


prompt_texts = ["Translate to German: My name is Arthur",
                "Please answer to the following question. Who is going to be the next Ballon d'or?",
                "Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.",
                "Please answer the following question. What is the boiling point of Nitrogen?",
                "Answer the following yes/no question. Can you write a whole Haiku in a single tweet?",
                "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?",
                "Q: ( False or not False or False ) is? A: Let's think step by step",
                "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?",
                "Premise: At my age you will probably have learnt one lesson. Hypothesis: It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?",
]

MODEL_CLASSES = [
    "t5",
]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_example_inputs(model):
    onnx_config_class = TasksManager.get_exporter_config_constructor(model_type=model.config.model_type, exporter="onnx", task="text2text-generation")
    onnx_config = onnx_config_class(model.config, use_past=model.config.use_cache)
    encoder_onnx_config = onnx_config.with_behavior("encoder")
    decoder_onnx_config = onnx_config.with_behavior("decoder", use_past=False)
    decoder_with_past_onnx_config = onnx_config.with_behavior("decoder", use_past=True)
    encoder_dummy_inputs = encoder_onnx_config.generate_dummy_inputs(framework="pt")
    decoder_dummy_inputs = decoder_onnx_config.generate_dummy_inputs(framework="pt")
    decoder_dummy_inputs["encoder_outputs"] = tuple(decoder_dummy_inputs["encoder_outputs"][0:1])
    decoder_with_past_dummy_inputs = decoder_with_past_onnx_config.generate_dummy_inputs(framework="pt")
    decoder_with_past_dummy_inputs["encoder_outputs"] = tuple(decoder_with_past_dummy_inputs["encoder_outputs"][0:1])
    decoder_with_past_dummy_inputs["past_key_values"] = tuple(decoder_with_past_dummy_inputs["past_key_values"])
    return encoder_dummy_inputs, decoder_dummy_inputs, decoder_with_past_dummy_inputs

#
# Functions to prepare models' input
#


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES),
    )
    parser.add_argument(
        "--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
    )
    parser.add_argument(
        "--dataset_cache_dir", nargs="?", default=None, const=None
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument(
        "--max-new-tokens", default=32, type=int, help="output max new tokens"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    parser.add_argument(
        "--output_dir",
        default="outputs_dir",
        type=str,
        help="Output directory where to save the resulting model",
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--ipex", action="store_true")
    parser.add_argument("--sq", action="store_true")
    parser.add_argument("--alpha", default="auto", help="Smooth quant parameter.")
    parser.add_argument(
        "--int8_bf16_mixed",
        action="store_true",
        help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--iters", default=100, type=int, help="num iter")
    parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--pad_max_length", default=512, type=int, help="Pad input ids to max length."
    )
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--calib_iters", default=100, type=int, help="calibration iters.")
    parser.add_argument("--tasks", nargs='+', default=["cnn_dailymail"], type=str, \
                        help="tasks list for accuracy validation")
    parser.add_argument("--save_accuracy_path", default=None,
                        help="Save accuracy results path.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.quantize or args.int8:
        model = INCModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.bf16: 
        model = INCModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, export=True, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    model.to(args.device)

    args.length = adjust_length_to_model(
        args.length,
        max_sequence_length=model.config.max_position_embeddings
        if hasattr(model.config, "max_position_embeddings")
        else 0,
    )
    logger.info(args)

    prefix = args.prefix if args.prefix else args.padding_text
    prompt_text = [prefix + args.prompt] if args.prompt else prompt_texts

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
            example = self.tokenizer(examples["text"])
            return example

        @torch.no_grad()
        def collate_batch(self, batch):
            input_ids_padded = []
            last_ind = []
            for text in batch:
                input_ids = text["input_ids"]
                pad_len = self.pad_max - input_ids.shape[0]
                last_ind.append(input_ids.shape[0] - 1)
                if self.is_calib:
                    input_ids = (
                        input_ids[: self.pad_max]
                        if len(input_ids) > self.pad_max
                        else input_ids
                    )
                else:
                    input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
                input_ids_padded.append(input_ids)
            return  (
                torch.vstack(input_ids_padded),
                torch.tensor(last_ind),
            )

    if args.quantize:
        encoder_dummy_inputs, decoder_dummy_inputs, decoder_with_past_dummy_inputs = get_example_inputs(model)
        from neural_compressor import PostTrainingQuantConfig, quantization

        excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
        calib_dataset = load_from_disk(args.dataset_cache_dir)["train"] if args.dataset_cache_dir else load_dataset(args.dataset, split="train")
        calib_dataset = calib_dataset.shuffle(seed=42)
        calib_evaluator = Evaluator(
            calib_dataset,
            tokenizer,
            args.batch_size,
            pad_max=args.pad_max_length,
            is_calib=True,
        )
        calib_size = 1
        calib_dataloader = DataLoader(
            calib_evaluator.dataset,
            batch_size=calib_size,
            shuffle=False,
            collate_fn=calib_evaluator.collate_batch,
        )

        def encoder_calib_func(prepared_model):
            for i, (input_ids, last_ind) in enumerate(calib_dataloader):
                input_bs, input_len = input_ids.shape
                attention_mask = torch.ones(input_bs, input_len)
                if i >= args.calib_iters:
                    break
                prepared_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        if args.sq:
            args.alpha = args.alpha if args.alpha == "auto" else float(args.alpha)
            recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": args.alpha}}
            encoder_conf = PostTrainingQuantConfig(
                backend="ipex" if args.ipex else "default",
                excluded_precisions=excluded_precisions,
                recipes=recipes,
                example_inputs=encoder_dummy_inputs,
            )
        else:
            encoder_conf = PostTrainingQuantConfig(
                backend="ipex" if args.ipex else "default",
                excluded_precisions=excluded_precisions,
                example_inputs=encoder_dummy_inputs,
            )
        model.encoder_model.config.return_dict = False
        encoder_model = quantization.fit(
            model.encoder_model,
            encoder_conf,
            calib_func=encoder_calib_func,
        )
        model.encoder_model = encoder_model
        print("=======encoder quantized=====")
        if args.ipex:
            def decoder_with_past_calib_func(prepared_model):
                model.decoder_with_past_model = prepared_model
                for i, (input_ids, last_ind) in enumerate(calib_dataloader):
                    input_bs, input_len = input_ids.shape
                    attention_mask = torch.ones(input_bs, input_len)
                    if i >= args.calib_iters:
                        break
                    model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                          temperature=0.9,
                          num_beams=4,
                        do_sample=False,
                    )
            if args.sq:
                args.alpha = args.alpha if args.alpha == "auto" else float(args.alpha)
                recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": args.alpha}}
                decoder_with_past_conf = PostTrainingQuantConfig(
                    backend="ipex" if args.ipex else "default",
                    excluded_precisions=excluded_precisions,
                    recipes=recipes,
                    example_inputs=decoder_with_past_dummy_inputs,
                )
            else:
                decoder_with_past_conf = PostTrainingQuantConfig(
                    backend="ipex" if args.ipex else "default",
                    excluded_precisions=excluded_precisions,
                    example_inputs=decoder_with_past_dummy_inputs,
                )
            model.decoder_model.config.return_dict = False
            decoder_with_past_model = quantization.fit(
                copy.deepcopy(model.decoder_model),
                decoder_with_past_conf,
                calib_func=decoder_with_past_calib_func,
            )
            model.decoder_with_past_model = decoder_with_past_model
            print("=======decoder_with_past quantized=====")
        def decoder_calib_func(prepared_model):
            model.decoder_model = prepared_model
            for i, (input_ids, last_ind) in enumerate(calib_dataloader):
                input_bs, input_len = input_ids.shape
                attention_mask = torch.ones(input_bs, input_len)
                if i >= args.calib_iters:
                    break
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.9,
                    num_beams=4,
                    do_sample=False,
                )
        if args.sq:
            args.alpha = args.alpha if args.alpha == "auto" else float(args.alpha)
            recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": args.alpha}}
            decoder_conf = PostTrainingQuantConfig(
                backend="ipex" if args.ipex else "default",
                excluded_precisions=excluded_precisions,
                recipes=recipes,
                example_inputs=decoder_dummy_inputs,
            )
        else:
            decoder_conf = PostTrainingQuantConfig(
                backend="ipex" if args.ipex else "default",
                excluded_precisions=excluded_precisions,
                example_inputs=decoder_dummy_inputs,
            )
        model.decoder_model.config.return_dict = False
        decoder_model = quantization.fit(
            copy.deepcopy(model.decoder_model),
            decoder_conf,
            calib_func=decoder_calib_func,
        )
        model.decoder_model = decoder_model
        print("=======decoder quantized=====")
        if args.ipex:
            model.config.torchscript = True
        model.config.torch_dtype = "int8"
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        inc_config = INCConfig(quantization=encoder_conf, save_onnx_model=False)
        inc_config.save_pretrained(args.output_dir)

    # Generation
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

    if args.ipex:
            import intel_extension_for_pytorch

    if args.benchmark:
        num_iter = args.iters
        length = len(prompt_text)
        # start
        total_time = 0.0
        total_token_num = 0
        num_warmup = args.num_warmup

        with torch.inference_mode(), torch.no_grad():
            for i in range(num_iter):
                input_size = tokenizer(prompt_text[i % length], return_tensors="pt").input_ids.size(dim=1)
                print("{}---- Prompt : {}, size: {}".format(i, prompt_text[i % length], input_size))

                tic = time.time()
                input_ids = tokenizer(prompt_text[i % length], return_tensors="pt").input_ids
                gen_ids = model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
                )
                gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                toc = time.time()
                output_tokens_num = gen_ids.numel()
                # Remove all text after the stop token
                gen_text = gen_text[: gen_text.find(args.stop_token) if args.stop_token else None]
                print(gen_text, flush=True)
                if i >= num_warmup:
                    total_time += toc - tic
                    total_token_num += output_tokens_num

        print("\n", "-" * 10, "Summary:", "-" * 10)
        latency = total_time / total_token_num
        print("Inference latency: %.3f sec/token." % latency)
        throughput = total_token_num / total_time
        print("Throughput: {} tokens/sec".format(throughput))

    if args.accuracy:
        from intel_extension_for_transformers.evaluation.hf_eval import summarization_evaluate
        results = summarization_evaluate(
           model=model,
           tokenizer_name=args.model_name_or_path,
           batch_size=1,
           limit=500,
        )
        dumped = json.dumps(results, indent=2)
        if args.save_accuracy_path:
            with open(args.save_accuracy_path, "w") as f:
                f.write(dumped)
        print("Accuracy (rouge2) for %s is: %s" % ("cnn_dailymail", results["rouge2"]))


if __name__ == "__main__":
    main()
