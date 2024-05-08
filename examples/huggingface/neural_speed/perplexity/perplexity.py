#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import argparse
import logging
import math
import os
import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger('perplexity')
'''
Preparing test dataset:

>>> import datasets
>>> dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split='test', num_proc=16)
>>> dataset.save_to_disk('~/wikitext-2-raw-v1-data-test')
>>> dataset = datasets.load_dataset("pg19", split='test', num_proc=16)
>>> dataset.save_to_disk('~/pg19-data-test')
'''


def try_resolve_dir(d):
    resolved = pathlib.Path(d).expanduser().resolve()
    if resolved.exists():
        return str(resolved)
    return d


def get_ppl(sum_nll, sum_nll2, cnt: int):
    ''' Get ppl and its standard deviation from sum of negative log likelihood '''
    nll = sum_nll / cnt
    nll2 = sum_nll2 / cnt
    ppl = math.exp(nll)
    return ppl, 0. if cnt <= 1 else math.sqrt((nll2 - nll * nll) / (cnt - 1))


def perplexity(model_name, dataset_name, **kwargs):
    import datasets
    from intel_extension_for_transformers.transformers import (AutoModelForCausalLM, RtnConfig)
    from transformers import AutoTokenizer, AutoConfig
    model_name = try_resolve_dir(model_name)
    dataset_name = try_resolve_dir(dataset_name)

    ctx_size = kwargs.get("ctx_size", 256)
    prompt_size = kwargs.get("prompt_size", ctx_size // 4)  # use one quarter as prompt
    n_threads = kwargs.get("n_threads", len(os.sched_getaffinity(0)))  # Note: linux only
    n_pred_per_sample = kwargs.get("n_pred_per_sample", ctx_size * 2)
    n_sampels = kwargs.get("n_sampels", 2)
    data_text_concat = kwargs.get("data_text_concat", "wikitext-2-raw-v1" in dataset_name)  # concat samples with `\n\n`
    default_model_kwargs = {"n_batch": ctx_size, "ctx_size": ctx_size, "n_keep": 4}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data = datasets.load_from_disk(dataset_name)
    test_text = data['text']
    if data_text_concat:
        test_text = ['\n\n'.join(test_text)]

    if n_sampels < 0:
        n_sampels = len(test_text)
    elif n_sampels > len(test_text):
        logger.warning(f"Try to eval {n_sampels} samples but there are only {len(test_text)} in the dataset!")
        n_sampels = len(test_text)

    test_ids = []
    with tqdm(total=n_sampels, desc="tokenizing") as pbar:
        length_needed = prompt_size + n_pred_per_sample
        for text in test_text:
            if len(test_ids) > n_sampels:
                break
            ids = tokenizer(text, return_tensors="pt", max_length=length_needed, truncation=True).input_ids
            if ids.shape.numel() >= length_needed:
                test_ids.append(ids)
                pbar.update(1)

    quantized_weight_path = kwargs.pop('quantized_weight_path', None)
    if quantized_weight_path:
        from neural_speed import Model
        model = Model()
        assert pathlib.Path(quantized_weight_path).is_file(), "Quantized weight not exist!"
        model.bin_file = quantized_weight_path
        model.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model.model_type = Model.get_model_type(model.config)
        model.tokenizer = tokenizer
    else:
        woq_kwargs = {
            k: kwargs[k]
            for k in kwargs
            if k in ['use_cache', 'compute_dtype', 'weight_dtype', 'scale_dtype', 'group_size', 'use_ggml']
        }
        woq_config = RtnConfig(**woq_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

    model_kwargs = {k: kwargs[k] for k in kwargs if k in ['n_keep', 'shift_roped_k', 'memory_dtype']}
    model_kwargs = {**default_model_kwargs, **model_kwargs}

    ppl_hist = [{} for _ in range(n_sampels)]  # ppl_hist[i_sample][end_pos] = ppl
    sum_nll = [0. for _ in range(n_sampels)]  # sum of negative log likelihood
    sum_nll2 = [0. for _ in range(n_sampels)]  # sum of nll square

    pbar = tqdm(range(n_pred_per_sample * n_sampels))
    for i in pbar:
        i_sample = i // n_pred_per_sample
        i_pred = i % n_pred_per_sample

        is_first = (i_pred == 0)

        begin_pos = 0 if is_first else i_pred + prompt_size - 1
        end_pos = i_pred + prompt_size
        cur_input = test_ids[i_sample][:, begin_pos:end_pos]
        cur_target: torch.Tensor = test_ids[i_sample][:, end_pos]
        out = model(cur_input, threads=n_threads, reinit=is_first, **model_kwargs)
        logsoftmax = torch.from_numpy(out).log_softmax(-1)
        nll = logsoftmax.take_along_dim(cur_target.view(-1, 1), 1)
        assert len(nll) == 1
        nll_v = -nll.flatten().tolist()[0]
        sum_nll[i_sample] += nll_v
        sum_nll2[i_sample] += nll_v * nll_v

        cur_ppl, cur_sd = get_ppl(sum_nll[i_sample], sum_nll2[i_sample], i_pred + 1)
        msg = f"Sample {i_sample + 1} / {n_sampels}; PPL = {cur_ppl:.4f} +/- {cur_ppl * cur_sd:.5f}"
        pbar.set_description(msg, False)
        ppl_hist[i_sample][end_pos] = cur_ppl

    return ppl_hist


def add_log_ppl_line(ax: plt.Axes, ppl_data: List[Dict[int, float]], label="log PPL"):
    """ Plot PPL and return xmax / ymax"""
    xs = []
    ys = []
    max_pos = max(max(d.keys()) for d in ppl_data)
    for i in range(max_pos + 1):
        ppls = [d[i] for d in ppl_data if i in d]
        if not ppls:
            continue
        xs.append(i)
        ys.append(math.log(sum(ppls) / len(ppls)))  # average over samples
    ax.plot(xs, ys, label=label)

    xmax = xs[torch.argmax(torch.tensor(ys)).item()]
    ymax = max(ys)
    return xmax, ymax, xs, ys


def draw_ppl(img_path: str, ppl_data: List[Dict[int, float]], ctx_size: int, model_title: str):
    fig, ax = plt.subplots()
    xmax, ymax, _, _ = add_log_ppl_line(ax, ppl_data)
    ax.annotate(f"max={ymax:.4f}", (xmax, ymax))

    ctx_line = ax.axvline(ctx_size, linestyle='--', color='r')
    ctx_line.set_label('KV Cache Size')
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Log Perplexity')
    ax.legend()

    ax.set_title(model_title)
    fig.suptitle("Language modeling perplexity")
    fig.savefig(img_path)

    print(f"Max PPL: {math.exp(ymax)}")
    return fig


def add_quant_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('quantize config')
    group.add_argument('--quantized_weight_path',
                       type=str,
                       help="path to quantized weight; other quant args will be ignored if specified",
                       default="")
    group.add_argument('--use_cache', action="store_true", help="Use local quantized model if file exists")
    group.add_argument(
        "--weight_dtype",
        choices=["int4", "int8"],
        help="Data type of quantized weight: int4/int8 (default: int4)",
        default="int4",
    )
    group.add_argument(
        "--alg",
        type=str,
        help="Quantization algorithm to use: sym/asym (default: sym)",
        default="sym",
    )
    group.add_argument("--group_size", type=int, help="Group size: Int (default: 32)", default=32)
    group.add_argument(
        "--scale_dtype",
        type=str,
        help="Data type of scales: bf16/fp32 (default: fp32)",
        default="fp32",
    )
    group.add_argument(
        "--compute_dtype",
        type=str,
        help="Data type of Gemm computation: int8/bf16/fp32 (default: int8)",
        default="int8",
    )
    group.add_argument(
        "--use_ggml",
        action="store_true",
        help="enable ggml for quantization and inference",
    )
    return group


def add_run_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('model run config')
    group.add_argument(
        "--n_keep",
        type=int,
        help="Number of tokens to keep from the initial prompt: Int (default: 0; -1 = all)",
        default=1,
    )
    group.add_argument(
        "--shift_roped_k",
        action="store_true",
        help="Use ring-buffer and thus do not re-computing after reaching ctx_size (default: False)",
    )
    group.add_argument("--memory_dtype",
                       type=str,
                       help="Data type of the kv memory",
                       choices=['f32', 'f16', 'auto'],
                       default="auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate perplexity for a model given a dataset")
    parser.add_argument('--model_name', type=str, default="~/Llama-2-7b-chat-hf")
    parser.add_argument('--dataset_name', type=str, default="~/pg19-data-test")
    parser.add_argument('--ctx_size', type=int, default=256)
    parser.add_argument('--prompt_size', type=int)
    parser.add_argument('--n_threads', type=int)
    parser.add_argument('--n_pred_per_sample', type=int)
    parser.add_argument('--n_sampels', type=int)
    parser.add_argument('--data_text_concat', action="store_true", default=None)
    parser.add_argument('--fig_path', type=str, default="out/ppl.png")
    add_quant_args(parser)
    add_run_args(parser)

    ns_args = parser.parse_args()
    args = vars(ns_args)
    args = {k: args[k] for k in args if args[k] is not None}

    pathlib.Path.mkdir(pathlib.Path("out"), exist_ok=True)
    ppl_data = perplexity(**args)

    # draw the graph
    job_name = f"{ns_args.model_name}-{ns_args.weight_dtype}"
    if ns_args.weight_dtype != 'fp32':
        job_name += f"-{ns_args.compute_dtype}-g{ns_args.group_size}"

    job_name += f"-keep{ns_args.n_keep}"
    draw_ppl(ns_args.fig_path, ppl_data, ns_args.ctx_size, job_name)

    # dump raw data
    import json
    with open('out/ppl_data.json', 'w') as f:
        json.dump(ppl_data, f, indent=2)
