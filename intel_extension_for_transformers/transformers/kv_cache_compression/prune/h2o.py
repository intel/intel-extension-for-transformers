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
import torch
import torch.nn as nn

from .base import KVPruner, PruneConfig

def local_heavy_hitter_mask(attn_weights, heavy_budget, no_padding_seq_length=None):

    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    if no_padding_seq_length is None:
        padding_length = 0
    else:
        padding_length = seq_length - no_padding_seq_length

    offset = torch.finfo(attn_weights.dtype).min
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    accumulated_attention_score = torch.sum(
        tmp_attn[:,:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
    accumulated_attention_score[:,:,heavy_budget+padding_length:] = 0
    accumulated_attention_score[:,:,:padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[:,:, padding_length:heavy_budget+padding_length,
                padding_length:heavy_budget+padding_length] = True
    for token_index in range(heavy_budget+padding_length, seq_length):

        tmp_attn_index = nn.functional.softmax(
            attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        _, tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget-1, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        mask_bottom_index[:,:, token_index] = True
        mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index

    mask_bottom = torch.tril(mask_bottom, diagonal=0)

    return mask_bottom


def get_hh_mask(heavy_budget_ratio, recent_budget_ratio, attn_weights, local=True):
    heavy_budget = int(heavy_budget_ratio * attn_weights.shape[-1])
    recent_budget = int(recent_budget_ratio * attn_weights.shape[-1])
    if heavy_budget > 0:
        # Default: No padding applied to input
        if local:
            mask_bottom = local_heavy_hitter_mask(attn_weights, heavy_budget, None)
        else:
            tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
            tmp_sum = torch.sum(tmp_attn, dim=-2)
            _, tmp_topk = tmp_sum.topk(k=heavy_budget, dim=-1)

            zeros = torch.zeros_like(tmp_sum, dtype=torch.bool)
            mask_bottom = zeros.scatter(-1, tmp_topk, True).unsqueeze(2)
            mask_bottom = mask_bottom.expand(mask_bottom.shape[0], mask_bottom.shape[1], attn_weights.shape[-2], mask_bottom.shape[-1])
    else:
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

    # Recent Mask
    ones = torch.ones_like(attn_weights, dtype=torch.bool)
    ones = torch.triu(ones, diagonal=-recent_budget)
    mask_bottom = torch.logical_or(mask_bottom, ones)
    # Combine h2o+recent and apply casual mask
    mask_bottom = torch.tril(mask_bottom, diagonal=0)

    return mask_bottom

class H2OConfig(PruneConfig):
    def __init__(
            self,
            heavy_ratio: float = None,
            recent_ratio: float = None,
            heavy_budget: int = None,
            recent_budget: int = None,
            h2o_min_seqlen: int = -1,
            real_drop: bool = True,
            mean: bool = False,
            local: bool = True
    ):
        super().__init__()
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.heavy_budget = heavy_budget
        self.recent_budget = recent_budget
        self.h2o_min_seqlen = h2o_min_seqlen
        self.real_drop = real_drop
        self.mean = mean
        self.local = local


class H2OKVCache:
    def __init__(
        self,
        heavy_ratio=0.2,
        recent_ratio=0.2,
        heavy_budget=None,
        recent_budget=None,
        min_seqlen=-1,
        mean=False
    ):
        ## bsz, num_heads, seq_len, head_dim
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.heavy_budget = heavy_budget
        self.recent_budget = recent_budget
        self.hh_score = None
        self.min_seqlen = min_seqlen
        self.mean = mean
        self.idx = 0

    def __call__(self, attn_score, key_states, value_states, **kwargs):
        seq_len = key_states.size(-2)
        if self.heavy_budget is None:
            self.heavy_budget = int(self.heavy_ratio * seq_len)
        if self.recent_budget is None:
            self.recent_budget = int(self.recent_ratio * seq_len)
        cache_size = self.heavy_budget + self.recent_budget
        if seq_len <= self.min_seqlen or seq_len <= cache_size:
            return key_states, value_states
        self.idx += 1
        # attn_score shape (bsz, num_heads, seq_len, head_dim)
        if len(attn_score.shape) == 3:
            attn_score = attn_score.unsqueeze(0)
        if len(attn_score.shape) == 5:
            attn_score = attn_score.reshape(
                attn_score.shape[0],
                attn_score.shape[1] * attn_score.shape[2],
                attn_score.shape[3],
                attn_score.shape[4]
                )
        self._update_hh_score(attn_score, mean=self.mean)

        # hh-selection
        mask = torch.zeros(self.hh_score.shape, dtype=attn_score.dtype).to(key_states.device)
        if not self.recent_budget == 0:
            mask[:,:,-self.recent_budget:] = 1 # pylint: disable=E1130
        select_hh_scores = self.hh_score[:,:,:seq_len - self.recent_budget]

        if not self.heavy_budget == 0:
            _, keep_topk = torch.topk(select_hh_scores, self.heavy_budget, dim=-1, largest=True)
            mask = mask.scatter(-1, keep_topk, 1)

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


class H2OKVPruner(KVPruner):
    def __init__(self, config: H2OConfig) -> None:
        self.config = config
        self.real_drop = self.config.real_drop
        self.prune_kv_cache_size = None


    def self_attn_init(self, module):
        module.h2o_kv_cache = H2OKVCache(
            self.config.heavy_ratio,
            self.config.recent_ratio,
            self.config.heavy_budget,
            self.config.recent_budget,
            self.config.h2o_min_seqlen,
            self.config.mean
            )

    def before_generate(self, model, inputs, *args, **kwargs):
        self.past_length = 0
        max_length = kwargs['max_new_tokens'] if kwargs.get('max_new_tokens') else kwargs['max_length']
        max_length += inputs.size(-1)
        for _, module in model.named_modules():
            if "Attention" in module.__class__.__name__:
                if module.h2o_kv_cache.heavy_budget is None:
                    module.h2o_kv_cache.heavy_budget = int(max_length * module.h2o_kv_cache.heavy_ratio)
                if module.h2o_kv_cache.recent_budget is None:
                    module.h2o_kv_cache.recent_budget = int(max_length * module.h2o_kv_cache.recent_ratio)
                if self.prune_kv_cache_size is None:
                    self.prune_kv_cache_size = module.h2o_kv_cache.recent_budget + module.h2o_kv_cache.heavy_budget

    def after_generate(self, model, inputs, *args, **kwargs):
        for _, module in model.named_modules():
            if "Attention" in module.__class__.__name__:
                module.h2o_kv_cache.clean_scores()
        self.prune_kv_cache_size = None

    def prune(self, module, query_states, key_states, value_states, causal_mask=None, **kwargs):
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(module.head_dim)
        if causal_mask is not None:  # no matter the length, we just slice it
            attn_weights = attn_weights + causal_mask
        if not self.config.real_drop:
            module.h2o_kv_cache.clean_scores()
        return module.h2o_kv_cache(attn_weights, key_states, value_states, **kwargs)

    def get_mask(self, module, query_states, key_states, value_states, causal_mask=None, **kwargs):
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(module.head_dim)
        if causal_mask is not None:  # no matter the length, we just slice it
            attn_weights = attn_weights + causal_mask
        mask = get_hh_mask(
            self.config.heavy_ratio,
            self.config.recent_ratio,
            attn_weights,
            local=self.config.local)
        return mask
