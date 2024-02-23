#!/usr/bin/env python
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

import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from .utils.nethook import Trace
from .utils.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

STATS_DIR = os.path.join(os.path.dirname(__file__), "stats")

def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    progress=tqdm,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = load_dataset(
            ds_name,
            # dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
        )
        field = "text"
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 512
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        ds = raw_ds["train"]
        if sample_size > len(raw_ds["train"]):
            ds = []
            for sample in raw_ds["train"]:
                splited_text = sample[field].split(' ')
                if len(splited_text) < maxlen:
                    ds.append({field:sample[field]})
                else:
                    for i in range(len(splited_text)//maxlen + 1):
                        partial_text = splited_text[i*maxlen:(i+1)*maxlen]
                        if partial_text:
                            ds.append({field:' '.join(partial_text)})
        return TokenizedDataset(ds, tokenizer, maxlen=maxlen, field=field)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    try:
        npos = model.config.n_positions
    except:
        npos = 512
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    stats_dir = Path(stats_dir)
    file_extension = \
        f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, model.device)
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat
