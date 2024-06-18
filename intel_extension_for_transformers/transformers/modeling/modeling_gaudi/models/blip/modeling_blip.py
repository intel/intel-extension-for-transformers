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

from typing import Optional

import torch
from transformers.utils import logging


logger = logging.get_logger(__name__)


@torch.no_grad()
def gaudi_BlipForConditionalGeneration_generate(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """
    The only differences are:
        - wrap hpu graph for each part
    """
    if generate_kwargs.get("hpu_graphs", True):
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph # pylint: disable=E0401 # pylint: disable=E0401

        if not hasattr(self.vision_model, "clear_cache"):
            self.vision_model = wrap_in_hpu_graph(self.vision_model)
        if not hasattr(self.text_decoder, "clear_cache"):
            self.text_decoder = wrap_in_hpu_graph(self.text_decoder)

    batch_size = pixel_values.shape[0]
    vision_outputs = self.vision_model(pixel_values=pixel_values)

    image_embeds = vision_outputs[0]

    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

    if isinstance(input_ids, list):
        input_ids = torch.LongTensor(input_ids)
    elif input_ids is None:
        input_ids = (
            torch.LongTensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
            .repeat(batch_size, 1)
            .to(image_embeds.device)
        )

    input_ids[:, 0] = self.config.text_config.bos_token_id
    attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

    outputs = self.text_decoder.generate(
        input_ids=input_ids[:, :-1],
        eos_token_id=self.config.text_config.sep_token_id,
        pad_token_id=self.config.text_config.pad_token_id,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        **generate_kwargs,
    )

    return outputs


@torch.no_grad()
def gaudi_BlipForQuestionAnswering_generate(
    self,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """
    The only differences are:
        - wrap hpu graph for each part
        - torch.full add dtype=torch.int64, or else the default type is torch.float32.
        lead to coredump in embedding layer
    """
    if generate_kwargs.get("hpu_graphs", True):
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph # pylint: disable=E0401 # pylint: disable=E0401

        if not hasattr(self.vision_model, "clear_cache"):
            self.vision_model = wrap_in_hpu_graph(self.vision_model)
        if not hasattr(self.text_encoder, "clear_cache"):
            self.text_encoder = wrap_in_hpu_graph(self.text_encoder)
        if not hasattr(self.text_decoder, "clear_cache"):
            self.text_decoder = wrap_in_hpu_graph(self.text_decoder)

    vision_outputs = self.vision_model(pixel_values=pixel_values)

    image_embeds = vision_outputs[0]

    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

    if isinstance(input_ids, list):
        input_ids = torch.LongTensor(input_ids)

    question_outputs = self.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        return_dict=False,
    )

    question_embeds = question_outputs[0]

    question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long).to(question_embeds.device)

    bos_ids = torch.full(
        (question_embeds.size(0), 1),
        fill_value=self.decoder_start_token_id,
        device=question_embeds.device,
        dtype=torch.int64,
    )

    outputs = self.text_decoder.generate(
        input_ids=bos_ids,
        eos_token_id=self.config.text_config.sep_token_id,
        pad_token_id=self.config.text_config.pad_token_id,
        encoder_hidden_states=question_embeds,
        encoder_attention_mask=question_attention_mask,
        **generate_kwargs,
    )

    return outputs
