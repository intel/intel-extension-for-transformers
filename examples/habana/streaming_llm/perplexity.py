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

"""
Adapted from https://github.com/tomaarsen/attention_sinks
"""


import itertools
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from optimum.habana.utils import get_hpu_memory_stats


def compute_perplexity(
    model,
    tokenizer,
    dataset,
    kv_window_size=1024,
    output_dir= "outputs",
    data_column= "text",
    num_samples = 1,
    num_tokens= None,
    overwrite= False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "attention_sink"
    output_file = output_dir / f"{suffix}.csv"

    if output_file.exists() and not overwrite:
        raise ValueError(
            f"The {output_file!r} output file already exists - if you really want to override it, then use `--overwrite`."
        )

    logs = defaultdict(list)
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    num_processed_tokens = 0

    # allocate kv cache
    model.allocate_kv_cache(1, kv_window_size, 1)
    for text in itertools.islice(dataset, num_samples):
        encodings = tokenizer(text[data_column], return_tensors="pt")

        seq_len = encodings.input_ids.size(1)
        print(f"sequence length: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            start_t = time.time()
            input_ids = encodings.input_ids[:, idx : idx + 1].to(model.device)
            pos_ids = torch.full((1,1), idx, dtype=torch.int64, device=model.device)
            with torch.no_grad():
                outputs = model(input_ids,
                                position_ids=pos_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                reuse_cache=True)
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                perplexity = neg_log_likelihood.exp()
            pbar.set_description(f"nll: {neg_log_likelihood.item():>5.2f}, ppl: {perplexity.item():>8.2f}")

            # Store data and save every 10 tokens
            logs["latency"].append(time.time() - start_t)
            logs["input_length"].append(idx + 1)
            logs["nll"].append(neg_log_likelihood.item())
            logs["ppl"].append(perplexity.item())
            logs["overall_ppl"].append(torch.tensor(logs["nll"]).mean().exp().item())
            logs["hpu_ram_allocated"].append(get_hpu_memory_stats().get('memory_allocated (GB)'))  # in GB
            if num_processed_tokens % 10 == 0:
                try:
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                except KeyboardInterrupt as ex:
                    # If there's a Keyboard Interrupt, still write the file, and then stop
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                    raise ex

            num_processed_tokens += 1
            if num_tokens and num_processed_tokens >= num_tokens:
                break
        if num_tokens and num_processed_tokens >= num_tokens:
                break

    pd.DataFrame(logs).to_csv(output_file, index=False)
    print(f"overall_ppl: {logs['overall_ppl'][-1]: >8.2f}")
    return
