#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import re
import torch
from typing import Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from optimum.intel.generation.modeling import TSModelForCausalLM
from intel_extension_for_transformers.transformers.utils.utility import (
    generate_dummy_past_key_values,
    generate_dummy_past_key_values_for_opt_llm,
    MODEL_TYPES_REQUIRING_POSITION_IDS,
)

ipex_opt_llm_supported = ["gptj", "opt", "llama", "gpt-neox", "falcon"]


class TSModelCausalLMForITREX(TSModelForCausalLM):
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.config.model_type == "bloom":
            return self._reorder_cache_bloom(past_key_values, beam_idx)
        if self.config.model_type == "chatglm":
            return tuple(
                tuple(
                    past_state.index_select(1, beam_idx.to(past_state.device))
                    for past_state in layer_past
                )
                for layer_past in past_key_values
            )

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        past_key_values = past_key_values or kwargs.get("past", None)

        if self.use_cache and past_key_values is not None:
            if not (
                self.config.model_type == "chatglm"
                and re.search("THUDM/chatglm-6b", self.config.auto_map["AutoConfig"])
            ):
                input_ids = input_ids[:, -1:]

        # `past_key_values` may be in the stardard format (e.g. in contrastive search),
        # converts to bloom's format if needed
        if past_key_values is not None and self.config.model_type == "bloom":
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)
        position_ids = kwargs.get("position_ids", None)

        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if self.config.model_type == "chatglm" and re.search(
            "THUDM/chatglm-6b", self.config.auto_map["AutoConfig"]
        ):
            MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
            seqs = input_ids.tolist()
            mask_positions, use_gmasks = [], []
            for seq in seqs:
                mask_token = gMASK if gMASK in seq else MASK
                use_gmask = mask_token == gMASK
                mask_positions.append(seq.index(mask_token))
                use_gmasks.append(use_gmask)
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            if past_key_values is None:
                context_lengths = [
                    seq.tolist().index(self.config.bos_token_id) for seq in input_ids
                ]
                position_ids = (
                    torch.arange(seq_length, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                )
                for i, context_length in enumerate(context_lengths):
                    position_ids[i, context_length:] = mask_positions[i]
                block_position_ids = [
                    torch.cat(
                        (
                            torch.zeros(
                                context_length, dtype=torch.long, device=device
                            ),
                            torch.arange(
                                seq_length - context_length,
                                dtype=torch.long,
                                device=device,
                            )
                            + 1,
                        )
                    )
                    for context_length in context_lengths
                ]
                block_position_ids = torch.stack(block_position_ids, dim=0)
                position_ids = torch.stack((position_ids, block_position_ids), dim=1)
            else:
                context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
                position_ids = torch.tensor(
                    [
                        [mask_position, seq_length - context_length]
                        for mask_position, context_length in zip(
                            mask_positions, context_lengths
                        )
                    ],
                    dtype=torch.long,
                    device=input_ids.device,
                ).unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": None,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        model_type = self.config.model_type.replace("_", "-")
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.use_cache and past_key_values is None:
            input_bs, input_len = input_ids.shape
            if model_type in ipex_opt_llm_supported:
                past_key_values = generate_dummy_past_key_values_for_opt_llm(
                    config=self.config, input_bs=input_bs, num_beams=1
                )
            else:
                past_key_values = generate_dummy_past_key_values(
                    config=self.config, input_bs=input_bs
                )
        inputs["past_key_values"] = past_key_values
        if attention_mask is None:
            inputs["attention_mask"] = torch.ones_like(input_ids)
        if model_type == "chatglm":
            inputs.pop("attention_mask")
            if re.search("THUDM/chatglm-6b", self.config.auto_map["AutoConfig"]):
                position_ids = self.prepare_inputs_for_generation(input_ids)[
                    "position_ids"
                ]

        if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
            if position_ids is not None:
                inputs["position_ids"] = position_ids
            else:
                inputs["position_ids"] = torch.arange(input_len).repeat(input_bs, 1)

        outputs = self.model(**inputs)

        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
            past_key_values = outputs[1] if self.use_cache else None
        else:
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"] if self.use_cache else None
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
