# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
###############################################################################
# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomMLP, dropout_add
from transformers.utils import logging

from ..modeling_attn_mask_utils import _gaudi_prepare_4d_causal_attention_mask


logger = logging.get_logger(__name__)


def gaudi_bloom_build_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype, training: bool
) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`):
            Number of heads.
        dtype (`torch.dtype`):
            Dtype of the output tensor.
        training (`bool`):
            Whether the model is being trained or not.
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    if training:
        arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
        alibi = slopes[..., None] * arange_tensor
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
    else:
        # code taken from Megatron transformer.py
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(seq_length, device=attention_mask.device).unsqueeze(
            0
        ).unsqueeze(0).expand(num_heads, -1, -1)

        # Select the part of the tensor that corresponds to our tensor parallel index.
        # if inference_tp_size is set use it instead of world size
        world = int(os.environ.get("WORLD_SIZE", 1))
        tp_world_size = GaudiBloomForCausalLM.inference_tp_size if GaudiBloomForCausalLM.inference_tp_size else world
        tp_index = 0  # if world size == 1 ignore rank and use 0 (for cases where WORLD_SIZE is not equal to tp size)
        if tp_world_size > 1:
            tp_index = int(os.environ.get("RANK", 0))

        alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

        alibi = alibi.repeat(batch_size, 1, 1)
        return alibi.to(dtype)


def update(prev, cur, dim, idx):
    if idx is not None:
        if os.environ.get("WA_INDEX_COPY", "1") == "1":
            past_selector, value_selector = idx
            if dim == 1:
                sel = torch.cat([past_selector, value_selector.unsqueeze(2)], dim=2)
                val = torch.cat([prev, cur], dim=1)
                return torch.bmm(sel, val)
            else:
                sel = torch.cat([past_selector, value_selector.unsqueeze(1)], dim=1)
                val = torch.cat([prev, cur], dim=2)
                return torch.bmm(val, sel)
        else:
            return prev.index_copy_(dim, idx - 1, cur)
    else:
        return torch.cat((prev, cur), dim=dim)


def gaudi_bloom_attention_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    token_idx: Optional[torch.Tensor] = None,
):
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, q_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

    # Collapse views to improve performance on HPU
    query_layer = query_layer.contiguous()
    key_layer = key_layer.contiguous()
    value_layer = value_layer.contiguous()

    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, head_dim, kv_length]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = update(past_key, key_layer, 2, token_idx)
        value_layer = update(past_value, value_layer, 1, token_idx)

    _, _, kv_length = key_layer.shape

    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None

    # [batch_size * num_heads, q_length, kv_length]
    # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
    matmul_result = alibi.baddbmm(
        batch1=query_layer,
        batch2=key_layer,
        beta=self.beta,
        alpha=self.inv_norm_factor,
    )

    # change view to [batch_size, num_heads, q_length, kv_length]
    attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

    # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
    input_dtype = attention_scores.dtype
    attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
    attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

    # [batch_size, num_heads, q_length, kv_length]
    attention_probs = self.attention_dropout(attention_probs)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    # change view [batch_size x num_heads, q_length, kv_length]
    attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

    # matmul: [batch_size * num_heads, q_length, head_dim]
    context_layer = torch.bmm(attention_probs_reshaped, value_layer)

    # change view [batch_size, q_length, num_heads * head_dim]
    context_layer = self._merge_heads(context_layer)

    # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + F.linear(
                context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
            )
    else:
        output_tensor = self.dense(context_layer)

    output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

    outputs = (output_tensor, present)
    if output_attentions:
        outputs += (attention_probs,)

    return outputs


class GaudiBloomMLP(BloomMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gelu_impl = torch.nn.GELU(approximate="tanh")


def gaudi_bloom_block_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    token_idx: Optional[torch.Tensor] = None,
):
    # hidden_states: [batch_size, seq_length, hidden_size]

    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.input_layernorm(hidden_states)

    # Layer norm post the self attention.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states

    # Self attention.
    attn_outputs = self.self_attention(
        layernorm_output,
        residual,
        layer_past=layer_past,
        attention_mask=attention_mask,
        alibi=alibi,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        token_idx=token_idx,
    )

    attention_output = attn_outputs[0]

    outputs = attn_outputs[1:]

    layernorm_output = self.post_attention_layernorm(attention_output)

    # Get residual
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = attention_output

    # MLP.
    output = self.mlp(layernorm_output, residual)

    if use_cache:
        outputs = (output,) + outputs
    else:
        outputs = (output,) + outputs[1:]

    return outputs  # hidden_states, present, attentions


def gaudi_bloom_convert_to_standard_cache(
    self, past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int, training: bool
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
    num_heads, ...]))"""
    batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
    if training:
        num_heads = batch_size_times_num_heads // batch_size
    else:
        world = int(os.environ.get("WORLD_SIZE", 1))
        tp_world_size = GaudiBloomForCausalLM.inference_tp_size if GaudiBloomForCausalLM.inference_tp_size else world
        num_heads = self.config.n_head // tp_world_size
        batch_size = batch_size_times_num_heads // num_heads
    # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
    # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
    return tuple(
        (
            layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
            layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
        )
        for layer_past in past_key_value
    )


def gaudi_bloom_convert_to_bloom_cache(
    self, past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))"""
    batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
    batch_size_times_num_heads = batch_size * num_heads
    # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
    # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
    return tuple(
        (
            layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
            layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
        )
        for layer_past in past_key_value
    )


def gaudi_bloom_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
    **deprecated_arguments,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
    if deprecated_arguments.pop("position_ids", False) is not False:
        # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
        warnings.warn(
            "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
            " passing `position_ids`.",
            FutureWarning,
        )
    if len(deprecated_arguments) > 0:
        raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    hidden_states = self.word_embeddings_layernorm(inputs_embeds)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # Compute alibi tensor: check gaudi_bloom_build_alibi_tensor
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values[0] is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
    else:
        attention_mask = attention_mask.to(hidden_states.device)

    alibi = gaudi_bloom_build_alibi_tensor(attention_mask, self.num_heads, hidden_states.dtype, self.training)

    causal_mask = _gaudi_prepare_4d_causal_attention_mask(
        attention_mask,
        input_shape=(batch_size, seq_length),
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )
    causal_mask = causal_mask.bool()

    if token_idx is not None and past_key_values[0] is not None and os.environ.get("WA_INDEX_COPY", "1") == "1":
        pkv = past_key_values[0][0]
        cur = torch.nn.functional.one_hot(torch.tile(token_idx - 1, (pkv.shape[0],)), pkv.shape[-1]).to(pkv.dtype)
        past = torch.diag_embed(1 - cur)
        token_idx = (past, cur)

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                alibi,
                causal_mask,
                layer_past,
                head_mask[i],
                use_cache,
                output_attentions,
                None,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
                token_idx=token_idx,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

    # Add last hidden state
    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


class GaudiBloomForCausalLM(BloomForCausalLM):
    inference_tp_size = None

    def set_tp_for_inference(tp_for_inference: int):
        world = int(os.environ.get("WORLD_SIZE", 1))
        assert tp_for_inference == 1 or tp_for_inference == world, "only setting 1 (no tp) or world size is supported"
        GaudiBloomForCausalLM.inference_tp_size = tp_for_inference

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last tokens for input_ids if past is not None
        if past_key_values is not None:
            if token_idx is None:
                input_ids = input_ids[:, -1].unsqueeze(-1)
            else:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)

            # the cache may be in the standard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "token_idx": token_idx,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""Labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):

        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx), training=self.training)

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)
