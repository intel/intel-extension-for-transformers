# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, StaticCache # pylint: disable=E0611
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    repeat_kv,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    logger,
)

from ...modeling.modeling_gaudi.models.modeling_attn_mask_utils import(
    _gaudi_prepare_4d_causal_attention_mask,
)


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE

    has_fused_rope = True
except ImportError:
    has_fused_rope = False
    print("Not using HPU fused kernel for apply_rotary_pos_emb")

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm

    has_fused_rms_norm = True
except ImportError:
    has_fused_rms_norm = False
    print("Not using HPU fused kernel for RMSNorm")

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

import habana_frameworks.torch.core as htcore # pylint: disable=E0401

from ..prune import PruneConfig, H2OConfig


def gaudi_llama_rmsnorm_forward(self, hidden_states):
    """

    The only differences are:
        - override RMSNorm with Habana fused RMSNorm
    """
    if hidden_states.device.type == "hpu" and has_fused_rms_norm:
        # mixed dtypes are not good for FusedRMSNorm, both inputs need to have same dtype
        if hidden_states.dtype != self.weight.dtype:
            orig_dtype = hidden_states.dtype
            hidden_states = FusedRMSNorm.apply(
                hidden_states.to(self.weight.dtype), self.weight, self.variance_epsilon
                )
            return hidden_states.to(orig_dtype)
        else:
            hidden_states = FusedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
            return hidden_states
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GaudiLlamaMLP(LlamaMLP):
    def pre_mlp_forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) \
                    for i in range(self.config.pretraining_tp)], dim=-1) # pylint: disable=E1102
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) \
                    for i in range(self.config.pretraining_tp)], dim=-1) # pylint: disable=E1102

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) \
                    for i in range(self.config.pretraining_tp)] # pylint: disable=E1102
            output = sum(down_proj)
        else:
            input = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            output = self.down_proj(input)
        return output

    def mlp_all_reduce(self, x):
        if hasattr(self.down_proj, "all_reduce"):
            self.down_proj.all_reduce(x)

    def post_mlp_forward(self, x):
        if self.config.pretraining_tp > 1:
            return x
        if hasattr(self.down_proj, "post_all_reduce"):
            return self.down_proj.post_all_reduce(x)
        return x


def gaudi_llama_repeat_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rep: int,
):
    """

    The only differences are:
        - Append num_key_value_heads == 1 check as kv states can be broadcasted during
          matmuls so need to expand and reshape them.
        - Add new args query_states, key_states, value_states and attention_mask and
          update the logic for expansion.
    The query states go from (batch, num_heads, seqlen, head_dim) to
    (batch, num_key_value_heads, n_rep, seqlen, head_dim)
    The key/value states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_key_value_heads, 1, seqlen, head_dim)
    """
    batch, num_key_value_heads, kv_len, head_dim = key_states.shape
    if n_rep == 1 or num_key_value_heads == 1:
        return query_states, key_states, value_states, attention_mask

    new_kv_shape = (batch, num_key_value_heads, 1, kv_len, head_dim)
    key_states = key_states.reshape(new_kv_shape)
    value_states = value_states.reshape(new_kv_shape)

    batch, _, q_len, head_dim = query_states.shape
    new_q_shape = (batch, num_key_value_heads, n_rep, q_len, head_dim)
    query_states = query_states.reshape(new_q_shape)

    if attention_mask is not None:
        # Add groups dim and set to 1
        attention_mask = attention_mask.unsqueeze(1)

    return query_states, key_states, value_states, attention_mask


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class KVCache(torch.nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            self.cache = torch.zeros(shape, dtype=dtype, device=device)
        else:
            assert (
                self.inp_seq_len == inp_seq_len
            ), f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
            self.cache.fill_(0)

    def update(self, prev, cur, dim, idx, inp_seq_len):
        orig_cur = cur
        if prev.shape == cur.shape:
            prev.copy_(cur)
            return orig_cur
        if cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
            # Initialize
            prev[:, :, :inp_seq_len, :].copy_(cur)
            return orig_cur
        assert cur.shape[2] == 1, f"Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur.shape}"
        if idx is not None:
            prev.index_copy_(dim, idx - 1, cur)
            return prev
        else:
            return torch.cat((prev, cur), dim=dim)

    def get_shape(self):
        if self.cache is None:
            return None
        return self.cache.shape

    def forward(self, cur, dim, idx):
        return self.update(self.cache, cur, dim, idx, self.inp_seq_len)


class GaudiLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self._cos_cached[:seq_len].to(dtype=x.dtype),
            self._sin_cached[:seq_len].to(dtype=x.dtype),
        )


class GaudiLlamaLinearScalingRotaryEmbedding(GaudiLlamaRotaryEmbedding):
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(dtype), persistent=False)


class GaudiLlamaDynamicNTKScalingRotaryEmbedding(GaudiLlamaRotaryEmbedding):
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(dtype), persistent=False)


class GaudiLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)

        self._init_func = []

    def register_init_func(self, func):
        self._init_func.append(func)

    def post_init(self):
        for func in self._init_func:
            func(self)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        cache_shape = (batch_size, self.num_key_value_heads, max_seq_len, self.head_dim)
        device = self.k_proj.weight.device
        dtype = self.config.torch_dtype
        self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
        self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)

    def update_sincos_cache(self, seq_len):
        # Call rotary emb forward() to update cos/sin cache when inferring more than self.max_position_embeddings
        # This helps in avoiding creation of these caches during actual model forward pass and
        # reduce memory consumption and improve performance.
        if seq_len > self.max_position_embeddings: # pylint: disable=E0203
            self.max_position_embeddings = seq_len
            _, _ = self.rotary_emb(self.k_proj.weight, seq_len=seq_len)

    def reorder(self, tensor, beam_idx, dim_a, dim_b):
        updated = tensor.index_select(0, beam_idx)
        tensor.copy_(updated)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        if self.k_cache.cache is None:
            return (None, None)

        head_dim = self.k_cache.cache.size(-1)
        seq_length = self.k_cache.cache.size(-2)
        self.reorder(self.k_cache.cache, beam_idx, seq_length, head_dim)
        self.reorder(self.v_cache.cache, beam_idx, seq_length, head_dim)
        return (self.k_cache.cache.shape, self.v_cache.cache.shape)

    def pre_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
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
        cache_prune_num: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """

        The only differences are:
        - add new args token_idx
        - optimize KV cache
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new arg flash_attention_causal_mask
        - add new arg cache_prune_num for attention_sinks
        """
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) \
                    for i in range(self.config.pretraining_tp)] # pylint: disable=E1102
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) \
                    for i in range(self.config.pretraining_tp)] # pylint: disable=E1102
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) \
                    for i in range(self.config.pretraining_tp)] # pylint: disable=E1102
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if token_idx is None:
                if hasattr(past_key_value, "get_usable_length"):
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                else:
                    kv_seq_len += past_key_value[0].shape[-2]
            else:
                if reuse_cache:
                    kv_seq_len = past_key_value[0][-2]
                else:
                    kv_seq_len = past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_customized_rope(query_states, key_states, cos, sin, position_ids)

        if use_cache:
            # reuse k, v, self_attention
            if reuse_cache:
                # key_states = self.k_cache(key_states, 2, token_idx)
                # value_states = self.v_cache(value_states, 2, token_idx)
                # past_key_value = (self.k_cache.get_shape(), self.v_cache.get_shape())

                # pruning kv cache
                if self.pruner.real_drop:
                    if attention_mask is not None:  # no matter the length, we just slice it
                        causal_mask = attention_mask
                        if cache_position is not None:
                            causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
                    if self.layer_idx == 0:
                        self.pruner.past_length += query_states.size(-2)
                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)
                    new_key_states, new_value_states = self.pruner.prune(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    causal_mask=causal_mask
                    )
                    new_key_states = self.pruner.remove_repeat_kv(new_key_states, self.num_key_value_groups)
                    new_value_states = self.pruner.remove_repeat_kv(new_value_states, self.num_key_value_groups)
                    key_states = self.k_cache(new_key_states, 2, token_idx)
                    value_states = self.v_cache(new_value_states, 2, token_idx)
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

        if use_flash_attention and FusedSDPA:
            import habana_frameworks.torch.hpu as ht # pylint: disable=E0401

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

            if not self.pruner.real_drop:
                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask
                    if cache_position is not None:
                        causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
                mask = self.pruner.get_mask(self, query_states, key_states, value_states,
                                        causal_mask=causal_mask)
                attn_weights[~mask] = torch.finfo(attn_weights.dtype).min

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
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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

    def attention_all_reduce(self, attn_output):
        if hasattr(self.o_proj, "all_reduce"):
            self.o_proj.all_reduce(attn_output)

    def post_attn_forward(self, attn_output):
        if hasattr(self.o_proj, "post_all_reduce"):
            self.o_proj.post_all_reduce(attn_output)
        return attn_output


class GaudiLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super(GaudiLlamaDecoderLayer, self).__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = GaudiLlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GaudiLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.self_attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.self_attn.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.self_attn.update_sincos_cache(seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        cache_idx: int = None,
        cache_prune_num: int = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """

        The only differences are:
        - add new args token_idx
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new arg flash_attention_causal_mask
        - add new arg cache_prune_num for attention_sinks
        """
        residual = hidden_states
        hidden_states, self_attn_weights, present_key_value = self.pre_attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            token_idx,
            attn_softmax_bf16,
            reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            cache_idx=cache_idx,
            cache_prune_num = cache_prune_num,
            **kwargs,
        )
        self.self_attn.attention_all_reduce(hidden_states)
        hidden_states, residual = self.post_attn_pre_mlp(hidden_states, residual)
        self.mlp.mlp_all_reduce(hidden_states)
        hidden_states = self.post_mlp(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def pre_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        cache_idx: int = None,
        cache_prune_num: int = 0,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present_key_value = self.self_attn.pre_attn_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            token_idx,
            attn_softmax_bf16,
            reuse_cache,
            use_flash_attention,
            flash_attention_recompute,
            flash_attention_causal_mask,
            cache_idx=cache_idx,
            cache_prune_num = cache_prune_num,
        )
        return hidden_states, attn_weights, present_key_value

    def post_attn_pre_mlp(self, hidden_states, residual):
        hidden_states = self.self_attn.post_attn_forward(hidden_states)

        if self.training:
            hidden_states = hidden_states + residual
            residual = hidden_states
        else:
            residual.add_(hidden_states)
            hidden_states = residual

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp.pre_mlp_forward(hidden_states)
        return hidden_states, residual

    def post_mlp(self, hidden_states, residual):
        hidden_states = self.mlp.post_mlp_forward(hidden_states)

        if self.training:
            hidden_states = hidden_states + residual
        else:
            residual.add_(hidden_states)
            hidden_states = residual

        return hidden_states


class GaudiLlamaModel(LlamaModel):
    """"""

    def __init__(self, config: LlamaConfig):
        """

        1. set fill_value to 1 instead of True
        2. add device=self.device
        """
        super(GaudiLlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [GaudiLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings),
            fill_value=1,
            dtype=torch.bool,
        )
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        for layer in self.layers:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return tuple(layer.reorder_kv_cache(beam_idx) for layer in self.layers)

    def update_sincos_cache(self, seq_len):
        for layer in self.layers:
            layer.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
        cache_prune_num: int = 0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        The only differences are:
        - add new args token_idx
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new arg flash_attention_causal_mask
        - add new arg lazy_mode
        - add new arg cache_prune_num for attention_sinks
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        ignore_cache_position = True  # Ignoring cache position for HPU
        use_new_cache = False  # Ignoring new Cache path for HPU
        past_seen_tokens = 0

        if past_key_values is not None and use_cache:  # kept for BC (cache positions)
            if reuse_cache:
                past_seen_tokens = past_key_values[0][0][2]
            else:
                if use_new_cache:
                    if not isinstance(past_key_values, StaticCache):
                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                    past_seen_tokens = past_key_values.get_seq_length()
                else:
                    past_seen_tokens = past_key_values[0][0].shape[2]

        if ignore_cache_position is False:
            if cache_position is None:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None and cache_position:
                position_ids = cache_position.unsqueeze(0)

        else:
            if position_ids is None:
                position_ids = torch.arange(
                    past_seen_tokens, seq_length + past_seen_tokens, dtype=torch.long, device=inputs_embeds.device
                )
                position_ids = position_ids.unsqueeze(0)
            cache_position = None

        # HPU specific mask generation
        if ignore_cache_position:
            # workaround for attention_sinks attention_mask which has fixed seq_len at dim -1
            if hasattr(self, "attention_sink_size") and hasattr(self, "attention_sink_window_size"):
                past_seen_tokens = self.attention_sink_size + self.attention_sink_window_size - seq_length
            causal_mask = _gaudi_prepare_4d_causal_attention_mask(
                attention_mask,
                input_ids.shape if input_ids is not None else (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if not use_new_cache else None

        if lazy_mode:
            htcore.mark_step()

        for layer_idx, decoder_layer in enumerate(self.layers):
            if (
                lazy_mode
                and not self.training
                and (torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1)
            ):
                htcore.mark_step()

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    None,
                    attn_softmax_bf16,
                    False,
                    use_flash_attention,
                    flash_attention_recompute,
                    flash_attention_causal_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None if past_key_values is None else past_key_values[layer_idx],
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    token_idx=token_idx,
                    attn_softmax_bf16=attn_softmax_bf16,
                    reuse_cache=reuse_cache,
                    use_flash_attention=use_flash_attention,
                    flash_attention_recompute=flash_attention_recompute,
                    flash_attention_causal_mask=flash_attention_causal_mask,
                    cache_idx=cache_idx,
                    cache_prune_num = cache_prune_num,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GaudiLlamaForCausalLM(LlamaPreTrainedModel):
    """

    The only differences are:
    - add new args token_idx
    - add token_idx into model_inputs
    - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
    - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_idx
    - add new args attn_softmax_bf16
    - add new args reuse_cache
    - add new arg cache_prune_num for attention_sinks
    """
    def __init__(
            self,
            config: LlamaConfig,
            prune_config: PruneConfig,
            ):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if isinstance(prune_config, H2OConfig):
            from ..prune import H2OKVPruner
            self.pruner = H2OKVPruner(prune_config)
        else:
            from ..prune import KVPruner
            self.pruner = KVPruner(prune_config)

        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            module = self.model.layers[layer_idx].self_attn
            self.model.layers[layer_idx].self_attn = GaudiLlamaAttention(
                config,
                layer_idx
                )
            self.model.layers[layer_idx].self_attn.register_init_func(self.pruner.self_attn_init)
            self.model.layers[layer_idx].self_attn.post_init()

            self.model.layers[layer_idx].self_attn.pruner = self.pruner

        # Initialize weights and apply final processing
        self.post_init()

        def _generate(*args, **kwargs):
            self.pruner.before_generate(self, *args, **kwargs)
            result = self.ori_generate(*args, **kwargs)
            self.pruner.after_generate(self,*args, **kwargs)
            return result

        self.ori_generate = self.generate
        self.generate = _generate

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.model.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.model.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.model.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        trim_logits: Optional[bool] = False,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
        cache_prune_num: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.generation_config.use_fused_rope is False:
            global has_fused_rope
            has_fused_rope = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            cache_idx=cache_idx,
            lazy_mode=lazy_mode,
            cache_prune_num=cache_prune_num,
        )
        hidden_states = outputs[0]
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) \
                    for i in range(self.config.pretraining_tp)] # pylint: disable=E1102
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, token_idx=None, **kwargs
    ):
        past_length = 0
        if not hasattr(self, "kv_past_token_length"):
            self.kv_past_token_length = 0
        using_attention_sinks = (hasattr(self, "attention_sink_size") and
                                 hasattr(self, "attention_sink_window_size"))

        reuse_cache = kwargs.get("reuse_cache")
        if past_key_values is not None:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            else:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                    max_cache_length = past_key_values.get_max_length()
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2]
                    max_cache_length = None

                # Keep only the unprocessed tokens:
                # 1 - If the length of the attention_mask exceeds the length of input_ids,
                # then we are in a setting where
                # some of the inputs are exclusively passed as part of the cache
                # (e.g. when passing input_embeds as input)
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                # 2 - If the past_length is smaller than input_ids',
                # then input_ids holds all input tokens. We can discard input_ids based on the past_length.
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                # 3 - Otherwise (past_length >= input_ids.shape[1]),
                # let's assume input_ids only has unprocessed tokens.

                # If we are about to go beyond the maximum cache length,
                # we need to crop the input attention mask.
                if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
                ):
                    attention_mask = attention_mask[:, -max_cache_length:] # pylint: disable=E1130
        elif reuse_cache and token_idx is not None:
            # With reuse_cache, KV cache is pre allocated hence for the 1st token
            # we can slice the inputs till token idx for the fwd pass
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]

        # prepare postion_ids and attention_mask for attention_sinks
        cache_prune_num = 0
        kv_cache_len = kwargs.get("kv_cache_len", None)
        position_ids = kwargs.get("position_ids", None)
        q_len = input_ids.shape[-1]
        if using_attention_sinks:
            assert (kv_cache_len and kv_cache_len == self.attention_sink_size + self.attention_sink_window_size)
            self.kv_past_token_length = min(self.kv_past_token_length, kv_cache_len)
            position_ids = torch.arange(self.kv_past_token_length,
                                        self.kv_past_token_length + q_len,
                                        device=input_ids.device)
            attn_sink_mask = torch.ones((q_len, kv_cache_len), device=input_ids.device)
            if self.kv_past_token_length < kv_cache_len:
                attn_sink_mask[:, self.kv_past_token_length:] = 0
            mask = torch.zeros((q_len, q_len), device=input_ids.device)
            mask_cond = torch.arange(mask.size(-1), device=mask.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
            if self.kv_past_token_length + q_len > kv_cache_len:
                cache_prune_num = (self.kv_past_token_length + q_len) - kv_cache_len
                position_ids = position_ids - cache_prune_num
            attn_sink_mask.index_copy_(-1, position_ids, mask)
            attention_mask= attn_sink_mask[None, None, :, :].expand(input_ids.shape[0], 1, q_len, kv_cache_len)
            position_ids = position_ids.unsqueeze(0)
        self.kv_past_token_length += q_len

        if attention_mask is not None and position_ids is None and not using_attention_sinks:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]
        # TODO: we are using token_idx, disable this for now
        # if self.generation_config.cache_implementation == "static":
        # generation with static cache
        # cache_position = kwargs.get("cache_position", None)
        # if cache_position is None:
        # past_length = 0
        # else:
        # past_length = cache_position[-1] + 1
        # input_ids = input_ids[:, past_length:]
        # position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        # cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)
        # keep cache_position implementation as None for HPU
        cache_position = None
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids.contiguous(),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "token_idx": token_idx,
                "trim_logits": kwargs.get("trim_logits"),
                "attn_softmax_bf16": kwargs.get("attn_softmax_bf16"),
                "reuse_cache": reuse_cache,
                "use_flash_attention": kwargs.get("use_flash_attention"),
                "flash_attention_recompute": kwargs.get("flash_attention_recompute"),
                "flash_attention_causal_mask": kwargs.get("flash_attention_causal_mask"),
                "cache_idx": kwargs.get("cache_idx"),
                "lazy_mode": kwargs.get("lazy_mode"),
                "cache_prune_num": cache_prune_num,
            }
        )
        return model_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Separate Attention Sink kwargs from regular kwargs
        attention_sink_kwargs = {key: value for key, value in kwargs.items() if key.startswith("attention_sink")}
        for key in attention_sink_kwargs:
            v = kwargs.pop(key)
            assert isinstance(v, int)

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        if len(attention_sink_kwargs) > 0:
            from intel_extension_for_transformers.transformers.modeling.modeling_gaudi.streaming_llm \
            import enable_streaming_llm

            enable_streaming_llm(model, **attention_sink_kwargs)
            model.attention_sink_size = attention_sink_kwargs.get("attention_sink_size")
            model.attention_sink_window_size = attention_sink_kwargs.get("attention_sink_window_size")
            model.model.attention_sink_size = attention_sink_kwargs.get("attention_sink_size")
            model.model.attention_sink_window_size = attention_sink_kwargs.get("attention_sink_window_size")

        return model


def apply_customized_rope(q, k, cos, sin, position_ids):
    if q.device.type == "hpu" and has_fused_rope:
        # TODO: remove `.clone()` when it is fixed in SynapseAI
        return FusedRoPE.apply(
            q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        ), FusedRoPE.apply(
            k, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        )
    else:

        return apply_rotary_pos_emb(q, k, cos[position_ids], sin[position_ids]) # pylint: disable=E1120
