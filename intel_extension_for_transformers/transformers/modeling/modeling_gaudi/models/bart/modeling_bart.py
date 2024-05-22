# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BART model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.utils import logging

from ..modeling_attn_mask_utils import (
    _gaudi_prepare_4d_causal_attention_mask,
)


logger = logging.get_logger(__name__)


# The difference is: modified dynamic shapes to static shapes with `mark_step` for performance improvement.


class gaudi_BartLearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: torch.Tensor = torch.tensor(0)):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device).expand(bsz, -1)
        positions += past_key_values_length

        return super().forward(positions + self.offset)


def gaudi_BartAttention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    token_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    # `past_key_value[0].shape[2] == key_value_states.shape[1]`
    # is checking that the `sequence_length` of the `past_key_value` is the same as
    # the provided `key_value_states` to support prefix tuning
    if is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        present_key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        present_value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if token_idx is not None:
            # HPU bug WA
            key_states, value_states = past_key_value
            key_states.index_add_(
                2, token_idx - 1, present_key_states - torch.index_select(key_states, 2, token_idx - 1)
            )
            value_states.index_add_(
                2, token_idx - 1, present_value_states - torch.index_select(value_states, 2, token_idx - 1)
            )
        else:
            key_states = torch.cat([past_key_value[0], present_key_states], dim=2)
            value_states = torch.cat([past_key_value[1], present_value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    value_states = value_states.reshape(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, past_key_value


def gaudi_BartEncoderLayer_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    layer_head_mask: torch.FloatTensor,
    output_attentions: Optional[bool] = False,
    token_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
    residual = hidden_states
    hidden_states, attn_weights, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        token_idx=token_idx,
    )
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def gaudi_BartDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = True,
    token_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    # Self Attention
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    # add present self-attn cache to positions 1,2 of present_key_value tuple
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        token_idx=token_idx,
    )
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    # Cross-Attention Block
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        if use_cache:
            cross_attn_past_key_value = (
                past_key_value[-2:] if (past_key_value is not None and past_key_value[-2:] != (None, None)) else None
            )
        else:
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            output_attentions=output_attentions,
            token_idx=token_idx,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

    # Fully Connected
    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def gaudi_BartEncoder_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutput]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input = input_ids
        input_ids = input_ids.view(-1, input_ids.shape[-1])
    elif inputs_embeds is not None:
        input = inputs_embeds[:, :, -1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

    embed_pos = self.embed_positions(input)
    import habana_frameworks.torch.core as htcore

    htcore.mark_step()
    embed_pos = embed_pos.to(inputs_embeds.device)

    hidden_states = inputs_embeds + embed_pos
    hidden_states = self.layernorm_embedding(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    # expand attention_mask
    if attention_mask is not None:
        if self._use_sdpa and head_mask is None and not output_attentions:
            # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
            # the manual implementation that requires a 4D causal mask in all cases.
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
        else:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        if head_mask.size()[0] != (len(self.layers)):
            raise ValueError(
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                f" {head_mask.size()[0]}."
            )

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        if to_drop:
            layer_outputs = (None, None)
        else:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                    output_attentions,
                    None,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                    token_idx=token_idx,
                )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


def gaudi_BartDecoder_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        input = input_ids
        input_shape = input.shape
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        input = inputs_embeds[:, :, -1]
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    tensor_past_key_values_length = token_idx - 1 if use_cache else torch.tensor(past_key_values_length)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input) * self.embed_scale

    if self._use_sdpa and not output_attentions and cross_attn_head_mask is None:
        # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _gaudi_prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

    # expand encoder attention mask
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        if self._use_sdpa and cross_attn_head_mask is None and not output_attentions:
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                encoder_attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1],
            )
        else:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

    # embed positions
    import habana_frameworks.torch.core as htcore

    htcore.mark_step()
    positions = self.embed_positions(input, tensor_past_key_values_length)

    htcore.mark_step()
    positions = positions.to(inputs_embeds.device)

    hidden_states = inputs_embeds + positions
    hidden_states = self.layernorm_embedding(hidden_states)

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    next_decoder_cache = () if use_cache else None

    # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:
                continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                head_mask[idx] if head_mask is not None else None,
                cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                None,
                output_attentions,
                use_cache,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_idx=token_idx,
            )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        cross_attentions=all_cross_attentions,
    )


def gaudi_BartModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    decoder_head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[List[torch.FloatTensor]] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[Tuple, Seq2SeqModelOutput]:
    # different to other models, Bart automatically creates decoder_input_ids from
    # input_ids if no decoder_input_ids are provided
    if decoder_input_ids is None and decoder_inputs_embeds is None:
        if input_ids is None:
            raise ValueError(
                "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                "passed, `input_ids` cannot be `None`. Please pass either "
                "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
            )

        decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if encoder_outputs is None:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
        )
    # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs[0],
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        token_idx=token_idx,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return Seq2SeqModelOutput(
        last_hidden_state=decoder_outputs.last_hidden_state,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def gaudi_BartForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    decoder_head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[List[torch.FloatTensor]] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[Tuple, Seq2SeqLMOutput]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None:
        if use_cache:
            logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
        use_cache = False
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

    outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask,
        decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        decoder_inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        token_idx=token_idx,
    )

    lm_logits = self.lm_head(outputs[0])
    lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

    masked_lm_loss = None
    if labels is not None:
        labels = labels.to(lm_logits.device)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return Seq2SeqLMOutput(
        loss=masked_lm_loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
    )


def gaudi_BartForConditionalGeneration_prepare_inputs_for_generation(
    self,
    decoder_input_ids,
    past_key_values=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    use_cache=None,
    encoder_outputs=None,
    token_idx=None,
    **kwargs,
):
    # cut decoder_input_ids if past_key_values is used
    if past_key_values is not None:
        if token_idx is not None:
            decoder_input_ids = torch.index_select(decoder_input_ids, 1, token_idx - 1)
        else:
            decoder_input_ids = decoder_input_ids[:, -1:].unsqueeze(-1)

    return {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        "token_idx": token_idx,
    }
