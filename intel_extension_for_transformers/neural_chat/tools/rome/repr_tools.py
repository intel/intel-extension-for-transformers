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

"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

import torch
from typing import List, Literal, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel

from .utils import nethook


def get_reprs_at_word_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: Optional[Literal["in", "out", "both"]] = "in",
    batch_first: Optional[bool] = True
) -> torch.Tensor:
    r"""
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tokenizer, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model=model,
        tokenizer=tokenizer,
        contexts=[context_templates[i].format(words[i]) for i in range(len(words))],
        idxs=idxs,
        layer=layer,
        module_template=module_template,
        track=track,
        batch_first=batch_first
    )


def get_words_idxs_in_templates(
    tokenizer: PreTrainedTokenizer, context_templates: List[str], words: List[str], subtoken: str
) -> List[List[int]]:
    r"""
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.

    We use left-padding so the words idxs are negative numbers.
    """

    assert all(tmp.count("{}") == 1 for tmp in context_templates), \
        "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    prefixes_len, words_len, suffixes_len, inputs_len = [], [], [], []
    for i, context in enumerate(context_templates):
        prefix, suffix = context.split("{}")
        prefix_len = len(tokenizer.encode(prefix))
        prompt_len = len(tokenizer.encode(prefix + words[i]))
        input_len = len(tokenizer.encode(prefix + words[i] + suffix))
        prefixes_len.append(prefix_len)
        words_len.append(prompt_len - prefix_len)
        suffixes_len.append(input_len - prompt_len)
        inputs_len.append(input_len)

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [prefixes_len[i] + words_len[i] - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0) - inputs_len[i]]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(len(context_templates))
        ]
    elif subtoken == "first":
        return [[prefixes_len[i] - inputs_len[i]] for i in range(len(context_templates))]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: Optional[Literal["in", "out", "both"]] = "in",
    batch_first: Optional[bool] = True
) -> torch.Tensor:
    r"""
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    if track == "both":
        to_return = {"in": [], "out": []}
    elif track == "in":
        to_return = {"in": []}
    elif track == "out":
        to_return = {"out": []}
    else:
        raise ValueError("invalid track")

    module_name = module_template.format(layer)

    def _batch(n): # batching to batches whose size is n
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    def _process(cur_repr, batch_idxs, key): # write results to return_dict
        nonlocal to_return
        cur_repr = cur_repr[0] if isinstance(cur_repr, tuple) else cur_repr
        if not batch_first: # (seq_len, batch_size, hidden_dim)
            cur_repr = cur_repr.transpose(0, 1)
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=512):
        contexts_tok = tokenizer(
            batch_contexts,
            padding=True,
            return_token_type_ids=False,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=("in" in to_return),
                retain_output=("out" in to_return),
            ) as tr:
                model(**contexts_tok)

        if "in" in to_return:
            _process(tr.input, batch_idxs, "in")
        if "out" in to_return:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if "in" in to_return else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
