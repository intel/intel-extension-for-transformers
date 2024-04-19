# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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


import os
import sys
import argparse
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import habana_frameworks.torch.core as htcore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intel_extension_for_transformers.transformers.modeling.modeling_gaudi import adapt_transformers_to_gaudi
from utils import print_memory_stats


def create_prompts(samples: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {"prompt": [prompt for prompts in samples["prompt"] for prompt in prompts]}


@torch.no_grad()
def greedy_generate(model, tokenizer, dataset, kv_cache=None, max_new_tokens=512, n_round=-1):
    streamer = TextStreamer(tokenizer)
    new_line_tokens = tokenizer("\n\n", return_tensors="pt", add_special_tokens=False).input_ids
    past_key_values = None
    num_token = 0
    count_round = 0

    for prompt_index, prompt in enumerate(dataset["prompt"]):
        # Use the chat template initially, as it adds the system prompt if the model has one, and then use [INST] and [/INST]
        if prompt_index:
            prompt = f"[INST] {prompt} [/INST]"
        else:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        num_token += input_ids.size(-1)

        streamer.put(input_ids)
        for _ in range(max_new_tokens):
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, reuse_cache=False)
            htcore.mark_step()
            past_key_values = outputs.past_key_values
            # print("past_key_values dtype:", past_key_values[0][0].dtype)
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            streamer.put(pred_token_idx)
            input_ids = pred_token_idx
            num_token += 1

            if pred_token_idx == tokenizer.eos_token_id:
                break

        streamer.put(new_line_tokens)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print_memory_stats()
        print("total token: {}k".format(num_token / 1000.0))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        count_round += 1
        if n_round > 0 and count_round >= n_round:
            break


def main():
    parser = argparse.ArgumentParser()
    # Model args
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--trust_remote_code", action="store_true")

    # Dataset args, not recommended to change:
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/mt_bench_prompts")
    parser.add_argument("--n_sample", type=int, default=-1)

    # Attention Sinks-only settings
    parser.add_argument("--enable_streaming", action="store_true")
    # Window size for attention_sinks
    parser.add_argument("--window_size", type=int, default=1024)
    # Attention Sink window size is calculated with args.window_size - args.attention_sink_size
    parser.add_argument("--attention_sink_size", type=int, default=4)

    # generation
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # optimize
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )

    args = parser.parse_args()
    args.quant_config = os.getenv("QUANT_CONFIG", "")

    # set hpu env
    # Tweak generation so that it runs faster on Gaudi
    adapt_transformers_to_gaudi()

    # Initialize the model
    if args.hf_token == "":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=bool(args.trust_remote_code),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=bool(args.trust_remote_code))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=bool(args.trust_remote_code),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=args.hf_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  token=args.hf_token,
                                                  trust_remote_code=bool(args.trust_remote_code))

     # Set up the kv cache
    if args.enable_streaming:
        from intel_extension_for_transformers.transformers.modeling.modeling_gaudi.streaming_llm import enable_streaming_llm
        kv_cache = enable_streaming_llm(model,
                                        attention_sink_size=args.attention_sink_size,
                                        window_size=args.window_size
        )
    else:
        from intel_extension_for_transformers.transformers.modeling.modeling_gaudi.models.llama.pos_shift_llama \
            import enable_gaudi_llama_cont_cat_kv_cache
        enable_gaudi_llama_cont_cat_kv_cache(model)
        kv_cache = None
        if args.quant_config or args.fp8:
            print("Error: can not use fp8 when not using attention sinks.")
            exit(0)

    # Set up model
    if args.fp8:
        import habana_frameworks.torch.core as htcore
        htcore.hpu_set_env()

    if args.quant_config:
        import habana_quantization_toolkit
        habana_quantization_toolkit.prep_model(model)

    model.eval().to("hpu")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        model = wrap_in_hpu_graph(model)
    if args.fp8:
        import habana_frameworks.torch.core as htcore
        print("Initializing inference mode")
        const_marking = os.getenv("ENABLE_CONST_MARKING", "True")
        if const_marking == "True":
            htcore.hpu_initialize(model)
    model.generation_config.use_flash_attention = False
    # print(model)
    print(model.generation_config)

    # Set up the dataset
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(create_prompts, batched=True, remove_columns=dataset.column_names)

    greedy_generate(model, tokenizer, dataset, kv_cache=kv_cache,
                    max_new_tokens=args.max_new_tokens, n_round=args.n_sample)

    if args.quant_config:
        import habana_quantization_toolkit
        habana_quantization_toolkit.finish_measurements(model)


if __name__ == "__main__":
    main()
