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
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding, GPTNeoXAttention, apply_rotary_pos_emb
# from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding


__all__ = ['convert_kvcache_gpt_neox_heavy_recent', 'GPTNeoXAttention_Mask']

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

    accumulated_attention_score = torch.sum(tmp_attn[:,:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
    accumulated_attention_score[:,:,heavy_budget+padding_length:] = 0
    if padding_length > 0:
        accumulated_attention_score[:,:,:padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[:,:, padding_length:heavy_budget+padding_length, padding_length:heavy_budget+padding_length] = True

    for token_index in range(heavy_budget+padding_length, seq_length):

        tmp_attn_index = nn.functional.softmax(attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        _, tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget-1, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        mask_bottom_index[:,:, token_index] = True

        mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index

    return mask_bottom

def sanity_check(mask):
    # mask (head, query, key)
    ones = torch.ones_like(mask)
    ones = torch.triu(ones, diagonal=0)
    mask_bottom = torch.logical_or(mask, ones)

    error_cnt = 0
    for i in range(mask_bottom.shape[1]-1):
        index = mask_bottom[:,i,:].eq(0).unsqueeze(1)
        index[:,i:]=0
        error_cnt += (mask_bottom[:,i:,:] * index).sum().item()
    print(error_cnt)
    return error_cnt

class GPTNeoXAttention_Mask(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        # self.rotary_emb = RotaryEmbedding(
        self.rotary_emb = GPTNeoXRotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        #attn_scores (bs, head, token, token)
        ### Heavy + Recent
        heavy_budget = int(self.heavy_budget_ratio * attn_scores.shape[-1])
        recent_budget = int(self.recent_budget_ratio * attn_scores.shape[-1])

        # Heavy Hitter Mask
        if heavy_budget > 0:
            mask_bottom = local_heavy_hitter_mask(attn_scores, heavy_budget, None) # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_scores, dtype=torch.bool)

        ones = torch.ones_like(attn_scores, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.logical_or(mask_bottom, ones)

        mask_bottom = torch.tril(mask_bottom, diagonal=0)

        # mask_bottom = ones
        attn_scores[~mask_bottom] = torch.finfo(attn_scores.dtype).min
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)

        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

def convert_kvcache_gpt_neox_heavy_recent(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_gpt_neox_heavy_recent(module, config)

        if isinstance(module, GPTNeoXAttention):
            model._modules[name] = GPTNeoXAttention_Mask(config)

    return model


