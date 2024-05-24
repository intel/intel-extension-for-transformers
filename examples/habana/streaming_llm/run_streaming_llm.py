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
from transformers import TextStreamer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_generation import setup_parser
from utils import print_memory_stats, initialize_model


def create_prompts(samples: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {"prompt": [prompt for prompts in samples["prompt"] for prompt in prompts]}


@torch.no_grad()
def greedy_generate(model, tokenizer, dataset, args, generation_config, max_new_tokens=512, n_round=-1):
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    new_line_tokens = tokenizer("\n\n", return_tensors="pt", add_special_tokens=False).input_ids
    num_token = 0
    count_round = 0

    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    generation_config.top_p = None
    generation_config.use_cache = True
    generation_config.attn_softmax_bf16 = True
    generation_config.reuse_cache = True
    generation_config.ignore_eos=False
    generation_config.bucket_size = -1
    generation_config.attention_sink_size = args.attention_sink_size
    generation_config.attention_sink_window_size = args.attention_sink_window_size
    print(generation_config)
    use_lazy_mode = True
    if args.torch_compile and model.config.model_type == "llama":
        use_lazy_mode = False
    for prompt_index, prompt in enumerate(dataset["prompt"]):
        if tokenizer.chat_template is not None:
            # Use the chat template initially, as it adds the system prompt if the model has one,
            # and then use [INST] and [/INST]
            if prompt_index:
                prompt = f"[INST] {prompt} [/INST]"
            else:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        outputs = model.generate(
            input_ids,
            generation_config=generation_config,
            streamer=streamer,
            lazy_mode=use_lazy_mode,
            hpu_graphs=args.use_hpu_graphs,
            profiling_steps=args.profiling_steps,
            profiling_warmup_steps=args.profiling_warmup_steps,
        ).cpu()

        # ignore padding token
        num_token += (outputs.shape[-1] - outputs[0].tolist().count(generation_config.pad_token_id))
        streamer.put(new_line_tokens)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print_memory_stats()
        print("total token: {}k".format(num_token / 1000.0))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        count_round += 1
        if n_round > 0 and count_round >= n_round:
            break

def setup_streaming_llm_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Streaming LLM script for HPU"
    )

    # Dataset args, not recommended to change
    # streaming demo: HuggingFaceH4/mt_bench_prompts
    # ppl: emozilla/pg19-test
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/mt_bench_prompts")
    parser.add_argument("--data_column", type=str, default="text")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--num_tokens", type=int, default=8192)

    # Window size for attention_sinks
    parser.add_argument("--attention_sink_window_size", type=int, default=1020)
    # Attention Sink whole window size is calculated with args.attention_sink_window_size + args.attention_sink_size
    parser.add_argument("--attention_sink_size", type=int, default=4)

    # compute perplexity and log
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--output_dir", type=str, default="benchmark/outputs")
    parser.add_argument("--overwrite", action="store_true")

    args = setup_parser(parser)

    return args

def main():
    args = setup_streaming_llm_parser()
    model, tokenizer, generation_config = initialize_model(args)

    if args.perplexity:  # compute perplexity
        from perplexity import compute_perplexity
        # Set up the dataset
        dataset = load_dataset(args.dataset, args.task, split=args.split, streaming=True)
        compute_perplexity(
            model,
            tokenizer,
            dataset,
            kv_window_size=args.attention_sink_window_size + args.attention_sink_size,
            output_dir=args.output_dir,
            data_column=args.data_column,
            num_samples=1,  # No support for more than one instance now
            num_tokens=args.num_tokens,
            overwrite=args.overwrite,
        )
    else:  # streaming generation demo
        # Set up the dataset
        dataset = load_dataset(args.dataset, split="train")
        dataset = dataset.map(create_prompts, batched=True, remove_columns=dataset.column_names)

        greedy_generate(model, tokenizer, dataset, args, generation_config,
                        max_new_tokens=args.max_new_tokens, n_round=args.num_sample)

    if args.quant_config:
        import habana_quantization_toolkit
        habana_quantization_toolkit.finish_measurements(model)


if __name__ == "__main__":
    main()
