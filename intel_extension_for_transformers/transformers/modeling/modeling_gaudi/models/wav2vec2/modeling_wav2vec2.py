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

from typing import Optional, Tuple, Union

import torch
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (
    BaseModelOutput,
    Wav2Vec2BaseModelOutput,
)


def _gaudi_wav2vec2_compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> torch.Tensor:
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L135
    The only difference is that the processing is performed with PyTorch on HPUs (Numpy is used in Transformers).
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = torch.rand([], device="hpu")

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device="hpu")
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = torch.randperm(input_length - (mask_length - 1), device="hpu")[:num_masked_span]

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = torch.cat(
            [
                spec_aug_mask_idx,
                torch.ones(max_num_masked_span - num_masked_span, dtype=torch.int32, device="hpu") * dummy_mask_idx,
            ]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx.to(dtype=torch.long))

    spec_aug_mask_idxs = torch.vstack(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = torch.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = torch.arange(mask_length, device="hpu")[None, None, :]
    offsets = torch.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    spec_aug_mask.scatter_(-1, spec_aug_mask_idxs, 1)

    return spec_aug_mask


def _gaudi_wav2vec2_sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[torch.Tensor] = None
):
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L254
    The only difference is that the processing is performed with PyTorch on HPUs (Numpy is used in Transformers).
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = torch.arange(sequence_length, device="hpu")

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = torch.zeros(
        shape=(batch_size, sequence_length, num_negatives), dtype=torch.int32, device="hpu"
    )

    mask_time_indices = (
        mask_time_indices.bool()
        if mask_time_indices is not None
        else torch.ones(features_shape, dtype=torch.bool, device="hpu")
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = torch.broadcast_to(torch.arange(high + 1, device="hpu")[:, None], (high + 1, num_negatives))
        sampled_indices = torch.randint(0, high, size=(high + 1, num_negatives), dtype=torch.int16, device="hpu")
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


def _gaudi_wav2vec2_mask_hidden_states(
    self,
    hidden_states: torch.FloatTensor,
    mask_time_indices: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
):
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1227
    Differences are that (1) `mask_time_indices` is not moved to the current device and converted into boolean because this is already done in _compute_mask_indices.
    (2) index_put operation on hidden_states is replaced by combination of simpler ops (more suitable for HPU graphs)
    """

    # `config.apply_spec_augment` can set masking to False
    if not getattr(self.config, "apply_spec_augment", True):
        return hidden_states

    # generate indices & apply SpecAugment along time axis
    batch_size, sequence_length, hidden_size = hidden_states.size()

    if mask_time_indices is not None:
        # apply SpecAugment along time axis with given mask_time_indices
        hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
    elif self.config.mask_time_prob > 0 and self.training:
        mask_time_indices = _gaudi_wav2vec2_compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=self.config.mask_time_prob,
            mask_length=self.config.mask_time_length,
            attention_mask=attention_mask,
            min_masks=self.config.mask_time_min_masks,
        )
        # replacement of index_put with combination of simpler ops. Assumption made about sizes of hidden_states (3d),
        # mask_time_indices (2d), self.masked_spec_embed (1d), for any other combination better to go back to original code using index_put.
        # hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        inverse_mask_time_indices = torch.bitwise_not(mask_time_indices)
        hidden_states = hidden_states * inverse_mask_time_indices.unsqueeze(2) + self.masked_spec_embed.to(
            hidden_states.dtype
        ).expand(hidden_states.size()) * mask_time_indices.unsqueeze(2)

    if self.config.mask_feature_prob > 0 and self.training:
        # generate indices & apply SpecAugment along feature axis
        mask_feature_indices = _gaudi_wav2vec2_compute_mask_indices(
            (batch_size, hidden_size),
            mask_prob=self.config.mask_feature_prob,
            mask_length=self.config.mask_feature_length,
            min_masks=self.config.mask_feature_min_masks,
        )
        mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
        mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
        hidden_states[mask_feature_indices] = 0

    return hidden_states


def gaudi_wav2vec2_encoder_forward(
    self,
    hidden_states: torch.tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    return_dict: bool = True,
):
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/7790943c91411f4234d11dfbf4c2f21ce7caf088/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L755
    The only difference is that torch.rand device is set to 'hpu' (required to capture operation as part of HPU graph)
    """
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    if attention_mask is not None:
        # make sure padded tokens output 0
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        hidden_states[~expand_attention_mask] = 0

        # extend attention_mask
        attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        )

    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

    for layer in self.layers:
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = torch.rand([], device="hpu")

        skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
        if not skip_the_layer or deepspeed_zero3_is_enabled:
            # under deepspeed zero3 all gpus must run in sync
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
            hidden_states = layer_outputs[0]

        if skip_the_layer:
            layer_outputs = (None, None)

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def gaudi_wav2vec2_forward(
    self,
    input_values: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    mask_time_indices: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1282
    The only difference is that a clone of `hidden_states` is given to _mask_hidden_states to avoid an error.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    extract_features = self.feature_extractor(input_values)
    extract_features = extract_features.transpose(1, 2)

    if attention_mask is not None:
        # compute reduced attention_mask corresponding to feature vectors
        attention_mask = self._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False
        )

    hidden_states, extract_features = self.feature_projection(extract_features)
    hidden_states = self._mask_hidden_states(
        hidden_states.clone(), mask_time_indices=mask_time_indices, attention_mask=attention_mask
    )

    encoder_outputs = self.encoder(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = encoder_outputs[0]

    if self.adapter is not None:
        hidden_states = self.adapter(hidden_states)

    if not return_dict:
        return (hidden_states, extract_features) + encoder_outputs[1:]

    return Wav2Vec2BaseModelOutput(
        last_hidden_state=hidden_states,
        extract_features=extract_features,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def gaudi_wav2vec2_tdnnlayer_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2290
    v4.38.2 implementation caused accuracy issue to run pytest Wav2Vec2RobustModelTest.
    """
    hidden_states = hidden_states.unsqueeze(1)
    hidden_states = torch.nn.functional.unfold(
        hidden_states,
        (self.kernel_size, self.in_conv_dim),
        stride=(1, self.in_conv_dim),
        dilation=(self.dilation, 1),
    )
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = self.kernel(hidden_states)

    hidden_states = self.activation(hidden_states)
    return hidden_states
