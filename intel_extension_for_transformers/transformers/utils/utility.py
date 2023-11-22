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

"""Utils for pytorch framework."""

import os
from typing import Optional, Tuple
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport
from transformers.modeling_outputs import CausalLMOutputWithPast


CONFIG_NAME = "best_configure.yaml"
ENGINE_MODEL_NAME = "model.bin"
ENGINE_MODEL_CONFIG = "conf.yaml"
ENCODER_NAME = "encoder_model.bin"
DECODER_NAME = "decoder_model.bin"
DECODER_WITH_PAST_NAME = "decoder_with_past_model.bin"
WEIGHTS_NAME = "pytorch_model.bin"

torch = LazyImport("torch")

def distributed_init(backend="gloo", world_size=1, rank=-1, init_method=None,
                     master_addr='127.0.0.1', master_port='12345'):
    """Init the distibute environment."""
    torch = LazyImport("torch")
    rank = int(os.environ.get("RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", world_size))
    if init_method is None:
        master_addr = os.environ.get("MASTER_ADDR", master_addr)
        master_port = os.environ.get("MASTER_PORT", master_port)
        init_method = 'env://{addr}:{port}'.format(addr=master_addr, port=master_port)
    torch.distributed.init_process_group(
        backend,
        init_method=init_method,
        world_size=world_size, 
        rank=rank
    )


def remove_label(input):
    if "labels" in input:  # for GLUE
        input.pop('labels')
    elif "start_positions" in input and "end_positions" in input:  # for SQuAD
        # pragma: no cover
        input.pop('start_positions')
        input.pop('end_positions')
    return input


def _build_inc_dataloader(dataloader):
    # transformer issue #1
    # for transformers 4.31.0: accelerate dataloader
    # *** ValueError: batch_size attribute should not be set 
    # after DataLoaderShard is initialized
    class INCDataLoader:
        __iter__ = dataloader.__iter__
        __len__ = dataloader.__len__
        def __init__(self) -> None:
            self.dataloader = dataloader
            self.batch_size = dataloader.total_batch_size
            self.dataset = dataloader.dataset
    return INCDataLoader()

def generate_dummy_past_key_values(config, input_bs):
    """
        Generate the dummy past_key_values.
    """
    from optimum.utils import NormalizedConfigManager
    normalized_config = NormalizedConfigManager.get_normalized_config_class(
        config.model_type
    )(config)
    nb_pkv = 2
    num_layers = normalized_config.num_layers
    num_attention_heads = normalized_config.num_attention_heads
    hidden_size = normalized_config.hidden_size
    d_k = hidden_size // num_attention_heads
    num_key_value_heads = num_attention_heads
    if hasattr(normalized_config, "num_key_value_heads"):
        num_key_value_heads = normalized_config.num_key_value_heads
    if hasattr(normalized_config, "multi_query_group_num"):
        num_key_value_heads = normalized_config.multi_query_group_num

    if config.model_type == "bloom":
        pkv = ()
        for nb_pkv in range(nb_pkv):
            if nb_pkv % 2 == 0:
                new_shape = [input_bs * num_key_value_heads, d_k, 0]
            else:
                new_shape = [input_bs * num_key_value_heads, 0, d_k]
            pkv = pkv + (torch.zeros(size=new_shape),)
            past_key_values = tuple(tuple(pkv) for _ in range(num_layers))
            return past_key_values
    elif config.model_type == "qwen":
        new_shape = [input_bs, 0, num_key_value_heads, d_k]
    elif config.model_type == "chatglm":
        new_shape = [0, input_bs, num_key_value_heads, d_k]
    else:
        new_shape = [input_bs, num_key_value_heads, 0, d_k]
    past_key_values = [(
            torch.zeros(size=new_shape).contiguous(),
            torch.zeros(size=new_shape).contiguous())
              for _ in range(num_layers)]
    return tuple(past_key_values)

def generate_dummy_past_key_values_for_opt_llm(config, input_bs, num_beams=1):
    """
        Generate the dummy past_key_values.
    """
    from optimum.utils import NormalizedConfigManager
    normalized_config = NormalizedConfigManager.get_normalized_config_class(
        config.model_type
    )(config)
    num_layers = normalized_config.num_layers
    num_attention_heads = normalized_config.num_attention_heads
    hidden_size = normalized_config.hidden_size
    d_k = hidden_size // num_attention_heads
    num_key_value_heads = num_attention_heads
    nb_pkv = 2
    if hasattr(normalized_config, "num_key_value_heads"):
        num_key_value_heads = normalized_config.num_key_value_heads
    if hasattr(normalized_config, "multi_query_group_num"):
        num_key_value_heads = normalized_config.multi_query_group_num
    if config.model_type == "bloom":
        for nb_pkv in range(nb_pkv):
            if nb_pkv % 2 == 0:
                new_shape = [input_bs * num_key_value_heads, d_k, 1]
            else:
                new_shape = [input_bs * num_key_value_heads, 1, d_k]
    elif config.model_type == "qwen":
        new_shape = [input_bs, 1, num_key_value_heads, d_k]
    elif config.model_type == "chatglm":
        new_shape = [1, input_bs, num_key_value_heads, d_k]
    else:
        new_shape = [input_bs, num_key_value_heads, 1, d_k]

    beam_idx_tmp = torch.zeros(
                (2048, int(input_bs * num_beams)), dtype=torch.long
            ).contiguous()
    past_key_values = [(torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
            torch.zeros(size=new_shape).contiguous(),
            torch.zeros(size=new_shape).contiguous(),
            beam_idx_tmp) for _ in range(num_layers)]
    return tuple(past_key_values)


def get_example_inputs_for_chatglm(model, quantization_config=None, return_type="dict"):
    """
        Generate the example_input for tracing, support models load from AutoModelForCausalLM.

    """
    query = "我该怎么办?"
    tokenizer = quantization_config.tokenizer
    if hasattr(tokenizer, "build_chat_inputs"):
        inputs = tokenizer.build_chat_inputs(query)
        eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                        tokenizer.get_command("<|observation|>")]
        inputs["eos_token_id"] = eos_token_id
        input_ids = inputs["input_ids"]
        input_bs, input_len = input_ids.shape
        attention_mask = torch.ones(input_bs, input_len + 1)
        attention_mask[:, 0] = 0
        past_key_values = generate_dummy_past_key_values(input_bs, model)
    elif hasattr(tokenizer, "build_prompt"):
        prompt = tokenizer.build_prompt(query)
        inputs = tokenizer([prompt], return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_bs, input_len = input_ids.shape
        attention_mask = torch.ones(input_bs, input_len + 1)
        attention_mask[:, 0] = 0
        past_key_values = generate_dummy_past_key_values(input_bs, model)
    else:
        inputs = tokenizer([query], return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_bs, input_len = input_ids.shape
        attention_mask = torch.ones(input_bs, input_len + 1)
        attention_mask[:, 0] = 0
        past_key_values = generate_dummy_past_key_values(input_bs, model)
    example_inputs = {
        "input_ids": input_ids,
        "past_key_values": tuple(past_key_values),
        "attention_mask": attention_mask
    }
    return example_inputs

from optimum.intel.generation.modeling import TSModelForCausalLM
class TSModelCausalLMForOPTLLM(TSModelForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        model_type = self.config.model_type.replace("_", "-")
        if self.use_cache:
            if past_key_values is None:
                nb_pkv = 2
                num_layers = self.normalized_config.num_layers
                d_k = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
                batch_size = input_ids.shape[0]
                input_len = input_ids.shape[1]  
                num_attention_heads = self.normalized_config.num_attention_heads
                num_key_value_heads = num_attention_heads
                if hasattr(self.normalized_config, "num_key_value_heads"):
                    num_key_value_heads = self.normalized_config.num_key_value_heads
                if hasattr(self.normalized_config, "multi_query_group_num"):
                    num_key_value_heads = self.normalized_config.multi_query_group_num
                elif self.config.model_type == "qwen":
                    new_shape = [batch_size, 1, num_key_value_heads, d_k]
                elif self.config.model_type == "chatglm":
                    new_shape = [1, batch_size, num_key_value_heads, d_k]
                else:
                    new_shape = [batch_size, num_key_value_heads, 1, d_k]

                beam_idx_tmp = torch.zeros((2048, int(batch_size)), dtype=torch.long)
                past_key_values = [(torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros(size=new_shape).contiguous(),
                        torch.zeros(size=new_shape).contiguous(),
                        beam_idx_tmp) for _ in range(num_layers)]
                past_key_values = tuple(past_key_values)
            inputs["past_key_values"] = past_key_values
        if model_type != "opt":
            if position_ids is not None:
                inputs["position_ids"] = position_ids
            else:
                inputs["position_ids"] = torch.arange(input_len).repeat(batch_size, 1)

        outputs = self.model(**inputs)

        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
            past_key_values = outputs[1] if self.use_cache else None
        else:
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"] if self.use_cache else None
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)