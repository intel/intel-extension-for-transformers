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
import importlib
from functools import partial

import torch
from torch import nn

def set_module(model, op_name, new_module):
    """Set module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.
        new_module (object): the input model.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    setattr(module, name_list[-1], new_module)

def get_module(model, op_name):
    """Get module from model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    return module

def clean_cache(model):
    for _, module in model.named_modules():
        if "H2O" in module.__class__.__name__:
            module.h2o_kv_cache.clean_scores()

def generate(model, **kwargs):
    for _, module in model.named_modules():
        if "H2O" in module.__class__.__name__:
            module.is_gen = True
    result = model.ori_generate(**kwargs)
    clean_cache(model)
    return result

def convert_model(
        model,
        heavy_ratio,
        recent_ratio,
        h2o_min_seqlen=1024,
        real_drop=True,
        is_gen=False,
        mean=False
        ):
    model_type = model.config.model_type
    device = model.device
    atten_layers = []
    for name, module in model.named_modules():
        if "Attention" in module.__class__.__name__:
            atten_layers.append(name)

    for layer_name in atten_layers:
        module = get_module(model, layer_name)
        h2o_cls = getattr(
            importlib.import_module(
                f".models.modeling_{model_type}",
                "intel_extension_for_transformers.transformers.modeling.kv_cache_compression"
                ),
            "H2O" + module.__class__.__name__)
        module = h2o_cls(
            module,
            model.config,
            heavy_ratio,
            recent_ratio,
            h2o_min_seqlen=h2o_min_seqlen,
            real_drop=real_drop,
            mean=mean
            )
        set_module(model, layer_name, module)
    model.clean_cache = lambda: clean_cache(model)
    model.ori_generate = model.generate
    model.generate = partial(generate, model)
    model = model.to(device)
    return model


def local_heavy_hitter_mask(attn_weights, heavy_budget, no_padding_seq_length=None):

    # attn_weights (head, query, keys) or (BS, head, query, keys)
    attn_shape_len = len(attn_weights.shape)
    assert attn_shape_len in [3,4], \
        "Wrong shape of attn_weights. Should be (head, query, keys) or (BS, head, query, keys)"
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    if no_padding_seq_length is None:
        padding_length = 0
    else:
        padding_length = seq_length - no_padding_seq_length

    offset = torch.finfo(attn_weights.dtype).min
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    if len(attn_weights.shape) == 3:
        accumulated_attention_score = torch.sum(
            tmp_attn[:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
        accumulated_attention_score[:,heavy_budget+padding_length:] = 0
        if padding_length > 0:
            accumulated_attention_score[:,:padding_length] = 0
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom[:,padding_length:heavy_budget+padding_length,
                    padding_length:heavy_budget+padding_length] = True
    else:
        accumulated_attention_score = torch.sum(
            tmp_attn[:,:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
        accumulated_attention_score[:,:,heavy_budget+padding_length:] = 0
        accumulated_attention_score[:,:,:padding_length] = 0

        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom[:,:, padding_length:heavy_budget+padding_length,
                    padding_length:heavy_budget+padding_length] = True
    for token_index in range(heavy_budget+padding_length, seq_length):

        if attn_shape_len == 3:
            tmp_attn_index = nn.functional.softmax(
                attn_weights[:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        else:
            tmp_attn_index = nn.functional.softmax(
                attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        _, tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget-1, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        if attn_shape_len == 3:
            mask_bottom_index[:, token_index] = True
            mask_bottom[:,token_index,:] = mask_bottom_index
        else:
            mask_bottom_index[:,:, token_index] = True
            mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index

    mask_bottom = torch.tril(mask_bottom, diagonal=0)

    return mask_bottom


def get_hh_mask(heavy_budget_ratio, recent_budget_ratio, attn_weights):
    heavy_budget = int(heavy_budget_ratio * attn_weights.shape[-1])
    recent_budget = int(recent_budget_ratio * attn_weights.shape[-1])
    if heavy_budget > 0:
        # Default: No padding applied to input
        mask_bottom = local_heavy_hitter_mask(attn_weights, heavy_budget, None)
    else:
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

    # Recent Mask
    ones = torch.ones_like(attn_weights, dtype=torch.bool)
    ones = torch.triu(ones, diagonal=-recent_budget)
    mask_bottom = torch.logical_or(mask_bottom, ones)
    # Combine h2o+recent and apply casual mask
    mask_bottom = torch.tril(mask_bottom, diagonal=0)

    return mask_bottom

class H2OKVCache:
    def __init__(
        self,
        heavy_ratio=0.2,
        recent_ratio=0.2,
        real_drop=False,
        min_seqlen=-1
    ):
        ## bsz, num_heads, seq_len, head_dim
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.real_drop = real_drop
        self.min_seqlen = min_seqlen
        self.idx = 0


    def __call__(self, attn_score, key_states, value_states, mean=False, **kwargs):
        seq_len = key_states.size(-2)
        heavy_budget = int(self.heavy_ratio * seq_len)
        recent_budget = int(self.recent_ratio * seq_len)
        cache_size = heavy_budget + recent_budget
        if seq_len <= self.min_seqlen or seq_len <= cache_size:
            if self.real_drop:
                return key_states, value_states
            else:
                return torch.ones(attn_score.shape[:-1], dtype=attn_score.dtype).to(key_states.device)
        self.idx += 1
        # attn_score shape (bsz, num_heads, seq_len, head_dim)
        if len(attn_score.shape) == 3:
            attn_score = attn_score.unsqueeze(0)
        self._update_hh_score(attn_score, mean=mean)

        # hh-selection
        mask = torch.zeros(self.hh_score.shape, dtype=attn_score.dtype).to(key_states.device)
        if not recent_budget == 0:
            mask[:,:,-recent_budget:] = 1
        select_hh_scores = self.hh_score[:,:,:seq_len - recent_budget]

        if not heavy_budget == 0:
            _, keep_topk = torch.topk(select_hh_scores, heavy_budget, dim=-1, largest=True)
            mask = mask.scatter(-1, keep_topk, 1)

        if not self.real_drop:
            return mask

        mask = mask.bool()
        self.hh_score = self.hh_score[mask].view(self.hh_score.shape[0], self.hh_score.shape[1], cache_size)

        # if use repeat_kv, need to reshape mask
        n_rep = mask.size(1) / key_states.size(1)
        if n_rep > 1:
            drop_mask = torch.tensor(
                [True if i % n_rep == 0 else False for i in range(0, mask.size(1))]
                ).repeat(mask.size(0), 1).to(mask.device)
            mask = mask[drop_mask].view(key_states.shape[:-1])

        k_hh_recent = key_states[mask].view(key_states.shape[0], key_states.shape[1], cache_size, -1)
        v_hh_recent = value_states[mask].view(value_states.shape[0], value_states.shape[1], cache_size, -1)

        return k_hh_recent, v_hh_recent

    def _update_hh_score(self, attn_score_cache, mean=False):
        # attn_score_cache (bsz, num_heads, seq_len, head_dim)
        # hh_score size (bsz, num_heads, head_dim)

        attn_score_cache = attn_score_cache.sum(-2)
        if self.hh_score is not None:
        # clean self.hh_score if not generation mode
            if attn_score_cache.size(-1) < self.hh_score.size(-1):
                self.clean_scores()
            if not mean:
                attn_score_cache[:, :, :self.hh_score.shape[-1]] += self.hh_score
            else:
                attn_score_cache[:,:,:self.hh_score.shape[-1]] = attn_score_cache[:,:,:self.hh_score.shape[-1]] \
                    * (self.idx - 1) + self.hh_score 
                attn_score_cache /= self.idx

        self.hh_score = attn_score_cache


    def clean_scores(self):
        self.idx = 0
        self.hh_score = None
