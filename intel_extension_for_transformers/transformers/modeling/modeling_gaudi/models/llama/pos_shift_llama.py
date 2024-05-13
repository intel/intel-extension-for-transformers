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


from typing import Optional, Tuple
import types

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from .modeling_llama import (GaudiLlamaAttention,
                             gaudi_llama_repeat_kv,
                             has_fused_rope,
                             FusedSDPA,
                             KVCache)
from transformers.models.llama.modeling_llama import rotate_half

__all__ = ["enable_gaudi_llama_pos_shift_attention", "enable_gaudi_llama_pos_shift_kv_cache"]


def gaudi_apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # TODO shape dimension check
    if x.device.type == "hpu" and has_fused_rope:
        from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE
        return FusedRoPE.apply(
            x, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        )
    else:
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed

def gaudi_llama_pos_shift_kv_cache_allocate(self, inp_seq_len, dtype, device, shape):
    assert (
        self.window_size > inp_seq_len
    ), f"inp_seq_len ({inp_seq_len}) must be the less than window_size ({self.window_size})."
    if self.cache is None:
        self.inp_seq_len = inp_seq_len
        bs, num_heads, seq_len, head_dim = shape
        sink_shape = (bs, num_heads, self.window_size, head_dim)
        self.cache = torch.zeros(sink_shape, dtype=dtype, device=device)
    else:
        self.inp_seq_len = inp_seq_len

def gaudi_llama_pos_shift_kv_cache_update(self, prev, cur, dim, idx, inp_seq_len):
    orig_cur = cur
    if not hasattr(self, "update_count"):
        self.update_count = 0
    self.update_count += 1
    if cur.shape[2] >= 1 and cur.shape[2] <= prev.shape[2] and self.update_count == 1:
        # Initialize
        prev[:, :, :inp_seq_len, :].copy_(cur)
        return orig_cur
    elif idx is not None:
        if idx  + cur.shape[2] <= self.window_size:
            prev[:, :, idx:idx + cur.shape[2], :] = cur
        else:
            prev[:, :, -cur.shape[2]:, :] = cur
        return prev
    else:
        return torch.cat((prev, cur), dim=dim)

def gaudi_llama_pos_shift_kv_cache_forward(self, cur, dim, idx, kv_past_seq_len):
    def prune_window(cache, attention_sink_size, prune_num):
        if cache is None:
            return None
        # kv_seq_len will not exceed window_size (first prune then concat)
        window_size = cache.shape[-2]
        if kv_past_seq_len + prune_num <= window_size:
            return cache
        return torch.cat(
            [
                cache[:, :, 0:attention_sink_size, ...],
                cache[:, :, attention_sink_size + prune_num:window_size, ...],
                cache[:, :, attention_sink_size: attention_sink_size + prune_num, ...],
            ],
                dim=2,
        )

    self.cache = prune_window(self.cache, self.attention_sink_size, cur.shape[2])
    return self.update(self.cache, cur, dim, idx, self.inp_seq_len)

def gaudi_llama_pos_shift_pre_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    token_idx: Optional[torch.Tensor] = None,
    attn_softmax_bf16: Optional[bool] = False,
    reuse_cache: Optional[bool] = False,
    use_flash_attention: Optional[bool] = False,
    flash_attention_recompute: Optional[bool] = False,
    flash_attention_causal_mask: Optional[bool] = False,
    cache_idx: int = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Copied from LlamaAttention.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - add new args token_idx
    - optimize KV cache
    - add new args attn_softmax_bf16
    - add new args reuse_cache
    - add new args use_flash_attention
    - add new arg flash_attention_recompute
    - add new arg flash_attention_causal_mask
    In streaming-llm (attention sinks), reuse_cache=False, self.k_cache=None, self.v_cache=None, and
    the token_idx will not be used inside the function.
    """
    bsz, q_len, _ = hidden_states.size()
    if not hasattr(self, "kv_past_total_tokens"):
        self.kv_past_total_tokens = 0

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # TODO: update when auto mp params is enabled in DeepSpeed (cf. https://github.com/HabanaAI/DeepSpeed/blob/94309c7b5dfc1a69858f5c9f25737b2f81a332a5/deepspeed/module_inject/replace_module.py#L440)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2] if self.kv_past_total_tokens == 0 else self.kv_cache_max_sl
    if past_key_value is not None:
        if token_idx is None:
            if reuse_cache:
                kv_seq_len = past_key_value[0][-2]
            elif hasattr(past_key_value, "get_usable_length"):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]
        else:
            if reuse_cache:
                kv_seq_len = past_key_value[0][-2]
            else:
                kv_seq_len = past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    ### Shift Pos: query pos is min(cache_size, idx)
    if reuse_cache:
        # shift position_ids
        # for next model.generate round
        if token_idx is not None and token_idx < self.kv_past_total_tokens:
            position_ids = torch.arange(self.kv_past_total_tokens,
                                        self.kv_past_total_tokens + q_len,
                                        device=position_ids.device).unsqueeze(0)
        # exceed cache window size
        if self.kv_past_total_tokens >= self.kv_cache_max_sl:
            position_ids =  position_ids - (
                                            self.kv_past_total_tokens - self.kv_cache_max_sl # truncate num
                                            + position_ids.shape[-1]                             # prune_num
                                            )
    query_states = gaudi_apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)

    if use_cache:
        # reuse k, v, self_attention
        if reuse_cache:
            kv_past_seq_len = min(self.kv_past_total_tokens, self.kv_cache_max_sl)
            key_states = self.k_cache(key_states, 2, self.kv_past_total_tokens, kv_past_seq_len)
            value_states = self.v_cache(value_states, 2, self.kv_past_total_tokens, kv_past_seq_len)
            past_key_value = (self.k_cache.get_shape(), self.v_cache.get_shape())
        else:
            if past_key_value is None:
                past_key = torch.zeros(key_states.shape, dtype=self.k_proj.weight.dtype, device=key_states.device)
                past_value = torch.zeros(
                    key_states.shape, dtype=self.k_proj.weight.dtype, device=key_states.device
                )
                past_key_value = (past_key, past_value)
            key_states = self.k_cache.update(past_key_value[0], key_states, 2, token_idx, self.inp_seq_len)
            value_states = self.v_cache.update(past_key_value[1], value_states, 2, token_idx, self.inp_seq_len)
            if token_idx is None:
                past_key_value = (key_states, value_states)

        if cache_idx is not None and q_len == 1:
            key_states = key_states[:, :, :cache_idx, :]
            value_states = value_states[:, :, :cache_idx, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :, :cache_idx]
            kv_seq_len = key_states.shape[-2]
    else:
        past_key_value = None

    if attention_mask is not None and self.kv_past_total_tokens > 0:
        # fill the former mask for multi-round model.generate
        updated_attention_mask = torch.full((attention_mask.shape[0], attention_mask.shape[1], q_len, kv_seq_len),
                                            0,
                                            dtype=attention_mask.dtype,
                                            device=attention_mask.device)
        if self.kv_past_total_tokens < kv_seq_len:
            updated_attention_mask[:, :, :, self.kv_past_total_tokens:] = torch.finfo(attention_mask.dtype).min
        first_token = (attention_mask.shape[-1] == attention_mask.shape[-2])
        if first_token:
            if self.kv_past_total_tokens + q_len <= kv_seq_len:
                updated_attention_mask[:, :, :, self.kv_past_total_tokens:
                                                self.kv_past_total_tokens + q_len] = attention_mask
            else:
                updated_attention_mask[:, :, :, -q_len:] = attention_mask
        else:
            if self.kv_past_total_tokens + q_len <= self.kv_cache_max_sl:
                updated_attention_mask[:, :, :, self.kv_past_total_tokens:
                                                self.kv_past_total_tokens + q_len] = 0
        attention_mask = updated_attention_mask

    if attention_mask is None and q_len == 1 and kv_seq_len > self.kv_past_total_tokens:
        attention_mask = torch.full((1, 1, 1, kv_seq_len),
                                    torch.finfo(query_states.dtype).min,
                                    dtype=query_states.dtype,
                                    device=query_states.device)
        attention_mask[:, :, :, 0:self.kv_past_total_tokens + q_len] = 0

    self.kv_past_total_tokens += q_len

    ### Shift Pos: key pos is the pos in cache
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    key_states = gaudi_apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

    # TODO it will accelerate inference?
    import habana_frameworks.torch.core as htcore
    htcore.mark_step()

    if use_flash_attention and FusedSDPA:
        import habana_frameworks.torch.hpu as ht

        if q_len == 1:
            # next token
            with ht.sdp_kernel(enable_recompute=False):
                attn_output = FusedSDPA.apply(
                    query_states, key_states, value_states, attention_mask, 0.0, False, None
                )
        else:
            # first token
            if flash_attention_causal_mask:
                # causal masking on first token requires inputs to be of the same length
                with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                    attn_output = FusedSDPA.apply(query_states, key_states, value_states, None, 0.0, True, None)
            else:
                with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                    attn_output = FusedSDPA.apply(
                        query_states, key_states, value_states, attention_mask, 0.0, False, None
                    )

    else:
        query_states, key_states, value_states, attention_mask = gaudi_llama_repeat_kv(
            query_states, key_states, value_states, attention_mask, self.num_key_value_groups
        )

        attn_weights = self.matmul_qk(query_states, key_states.transpose(-2, -1)) * self.norm_factor

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if attn_softmax_bf16:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
        else:
            # upcast attention to fp32
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
        attn_output = self.matmul_av(attn_weights, value_states)
        attn_output = attn_output.reshape(bsz, -1, q_len, self.head_dim)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def enable_gaudi_llama_pos_shift_attention(model, max_attention_window_size):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gaudi_llama_pos_shift_attention(
                module, max_attention_window_size
            )

        if isinstance(module, GaudiLlamaAttention):
            model._modules[name].pre_attn_forward = types.MethodType(
                gaudi_llama_pos_shift_pre_attn_forward, model._modules[name]
            )
            model._modules[name].kv_cache_max_sl = max_attention_window_size

def enable_gaudi_llama_pos_shift_kv_cache(model, attention_sink_size, window_size):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gaudi_llama_pos_shift_kv_cache(
                module, attention_sink_size, window_size
            )

        if isinstance(module, KVCache):
            model._modules[name].allocate = types.MethodType(
                gaudi_llama_pos_shift_kv_cache_allocate, model._modules[name]
            )
            model._modules[name].update = types.MethodType(
                gaudi_llama_pos_shift_kv_cache_update, model._modules[name]
            )
            model._modules[name].forward = types.MethodType(
                gaudi_llama_pos_shift_kv_cache_forward, model._modules[name]
            )
            model._modules[name].attention_sink_size = attention_sink_size
            model._modules[name].window_size = window_size
