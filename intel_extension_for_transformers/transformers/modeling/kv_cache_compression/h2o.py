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
import math
from collections import OrderedDict
import importlib

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

def convert_model(
        model,
        heavy_ratio,
        recent_ratio,
        h2o_min_seqlen=1024,
        real_drop=True,
        is_gen=False
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
        module = h2o_cls(module, module.config, heavy_ratio, recent_ratio, h2o_min_seqlen, real_drop, is_gen)
        set_module(model, layer_name, module)
    model = model.to(device)
    return model


def local_heavy_hitter_mask(attn_weights, heavy_budget, no_padding_seq_length=None):

    # attn_weights (head, query, keys) or (BS, head, query, keys)
    attn_shape_len = len(attn_weights.shape)
    assert attn_shape_len in [3,4], "Wrong shape of attn_weights. Should be (head, query, keys) or (BS, head, query, keys)"
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    if no_padding_seq_length is None:
        padding_length = 0
    else:
        padding_length = seq_length - no_padding_seq_length

    offset = torch.finfo(attn_weights.dtype).min
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    if len(attn_weights.shape) == 3:
        accumulated_attention_score = torch.sum(tmp_attn[:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
        accumulated_attention_score[:,heavy_budget+padding_length:] = 0
        if padding_length > 0:
            accumulated_attention_score[:,:padding_length] = 0
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom[:, padding_length:heavy_budget+padding_length, padding_length:heavy_budget+padding_length] = True
    else:
        accumulated_attention_score = torch.sum(tmp_attn[:,:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
        accumulated_attention_score[:,:,heavy_budget+padding_length:] = 0
        accumulated_attention_score[:,:,:padding_length] = 0

        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom[:,:, padding_length:heavy_budget+padding_length, padding_length:heavy_budget+padding_length] = True
    for token_index in range(heavy_budget+padding_length, seq_length):

        if attn_shape_len == 3:
            tmp_attn_index = nn.functional.softmax(attn_weights[:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        else:
            tmp_attn_index = nn.functional.softmax(attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
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
        mask_bottom = local_heavy_hitter_mask(attn_weights, heavy_budget, None) # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

    # Recent Mask
    ones = torch.ones_like(attn_weights, dtype=torch.bool)
    ones = torch.triu(ones, diagonal=-recent_budget)
    mask_bottom = torch.logical_or(mask_bottom, ones)
    # Combine h2o+recent and apply casual mask
    mask_bottom = torch.tril(mask_bottom, diagonal=0)

    # ones = torch.ones_like(attn_weights, dtype=torch.bool)
    # ones = torch.tril(ones, diagonal=recent_budget)
    # ones = torch.triu(ones, diagonal=-recent_budget)
    # mask_bottom = torch.logical_or(mask_bottom, ones)
    return mask_bottom

class H2OKVCache:
    def __init__(
        self,
        heavy_ratio=0.2,
        recent_ratio=0.2,
        real_drop=False,
        min_seqlen=-1
    ):
        ## bsz, num_heads, seq_len, head_dim | num_heads, seq_len, head_dim
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.real_drop = real_drop
        self.min_seqlen = min_seqlen
        self.idx = 0


    def __call__(self, attn_score, key_states, value_states, mean=False, **kwargs):
        if key_states.shape[-2] <= self.min_seqlen:
            if self.real_drop:
                return key_states, value_states
            else:
                return torch.ones(self.hh_score.shape, dtype=attn_score.dtype).to(key_states.device)
        self.idx += 1
        self._update_hh_score(attn_score, mean=mean)
        heavy_budget = int(self.heavy_ratio * self.hh_score.shape[-1])
        recent_budget = int(self.recent_ratio * self.hh_score.shape[-1])
        cache_size = heavy_budget + recent_budget

        seq_len = key_states.size(-2)
        if seq_len <= cache_size:
            return key_states, value_states

        # hh-selection
        mask = torch.zeros(self.hh_score.shape, dtype=attn_score.dtype).to(key_states.device)
        if len(self.hh_score.shape) == 2:
            if not recent_budget == 0:
                mask[:,-recent_budget:] = 1 
            select_hh_scores = self.hh_score[:, :seq_len - recent_budget]
        else:
            if not recent_budget == 0:
                mask[:,:,-recent_budget:] = 1 
            select_hh_scores = self.hh_score[:,:,:seq_len - recent_budget]

        if not heavy_budget == 0:
            _, keep_topk = torch.topk(select_hh_scores, heavy_budget, dim=-1, largest=True)
            mask = mask.scatter(-1, keep_topk, 1)

        if not self.real_drop:
            return mask

        mask = mask.bool()
        states_shape = list(key_states.shape)
        states_shape[-2] = cache_size
        k_hh_recent = key_states[mask].view(*states_shape)
        states_shape[-1] = -1
        v_hh_recent = value_states[mask].view(*states_shape)

        tmp_shape = list(self.hh_score.shape)
        tmp_shape[-1] = cache_size
        self.hh_score = self.hh_score[mask].view(*tmp_shape)
        return k_hh_recent, v_hh_recent

    def _update_hh_score(self, attn_score_cache, mean=False):
        # hh_score size (bsz, num_heads, seq_len) or (num_heads, seq_len)

        attn_score_cache = attn_score_cache.sum(-2)
        if self.hh_score is not None:
        # clean self.hh_score if not generation mode
            if attn_score_cache.shape[-1] < self.hh_score:
                self.clean_scores()
            elif len(attn_score_cache.shape) == 2:
                attn_score_cache[:, :self.hh_score.shape[-1]] += self.hh_score / (1 if not mean else self.idx)
            else:
                attn_score_cache[:, :, :self.hh_score.shape[-1]] += self.hh_score / (1 if not mean else self.idx)

        self.hh_score = attn_score_cache


    def clean_scores(self):
        self.hh_score = None
