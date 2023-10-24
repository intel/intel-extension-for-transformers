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

import importlib
import os
from neural_compressor.utils.utility import LazyImport
from neural_compressor.utils import logger


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

def generate_dummy_past_key_values(input_bs, model):
    """
        Generate the dummy past_key_values.
    """
    from optimum.utils import NormalizedConfigManager
    normalized_config = NormalizedConfigManager.get_normalized_config_class(
        model.config.model_type
    )(model.config)
    nb_pkv = 2
    num_layers = normalized_config.num_layers
    num_attention_heads = normalized_config.num_attention_heads
    hidden_size = normalized_config.hidden_size
    d_k = hidden_size // num_attention_heads

    if model.config.model_type == "bloom":
        pkv = ()
        for nb_pkv in range(nb_pkv):
            if nb_pkv % 2 == 0:
                new_shape = [input_bs * num_attention_heads, d_k, 1]
            else:
                new_shape = [input_bs * num_attention_heads, 1, d_k]
            pkv = pkv + (torch.ones(size=new_shape),)
    elif model.config.model_type == "qwen":
        new_shape = [input_bs, 1, num_attention_heads, d_k]
        dummy_tensor = torch.ones(size=new_shape)
        pkv = tuple(dummy_tensor for _ in range(nb_pkv))
    else:
        new_shape = [input_bs, num_attention_heads, 1, d_k]
        dummy_tensor = torch.ones(size=new_shape)
        pkv = tuple(dummy_tensor for _ in range(nb_pkv))
    past_key_values = tuple(tuple(pkv) for _ in range(num_layers))
    return past_key_values

def generate_dummy_past_key_values_for_optimize_transformers(input_bs, model, num_beams):
    """
        Generate the dummy past_key_values.
    """
    from optimum.utils import NormalizedConfigManager
    normalized_config = NormalizedConfigManager.get_normalized_config_class(
        model.config.model_type
    )(model.config)
    num_layers = normalized_config.num_layers
    num_attention_heads = normalized_config.num_attention_heads
    hidden_size = normalized_config.hidden_size
    d_k = hidden_size // num_attention_heads
    new_shape = [input_bs, num_attention_heads, 1, d_k]
    dummy_tensor = torch.zeros(size=new_shape).contiguous()
    beam_idx_tmp = torch.zeros((2048, int(input_bs * num_beams)), dtype=torch.long)
    past_key_values = [(torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
            dummy_tensor,
            dummy_tensor,
            beam_idx_tmp) for _ in range(num_layers)]
    return past_key_values

def get_example_inputs(model, return_type="tuple"):
    """
        Generate the example_input for tracing, support models load from AutoModelForCausalLM.

    """
    input_ids = model.dummy_inputs["input_ids"]
    input_bs, input_len = input_ids.shape
    past_key_values = generate_dummy_past_key_values(input_bs, model)
    attention_mask = torch.ones(input_bs, input_len + 1)
    attention_mask[:, 0] = 0
    example_inputs = (input_ids, tuple(past_key_values), attention_mask)
    # do inference to check example_inputs formats
    model(*example_inputs)
    if return_type != "tuple":
        example_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": tuple(past_key_values),
        }
    return example_inputs

def get_example_inputs_for_optimize_transformers(model, num_beams=4, return_type="tuple"):
    """
        Generate the example_input for tracing, support models load from AutoModelForCausalLM.

    """
    input_ids = model.dummy_inputs["input_ids"]
    input_bs, input_len = input_ids.shape
    past_key_values = generate_dummy_past_key_values_for_optimize_transformers(input_bs, model, num_beams)
    attention_mask = torch.ones(input_bs, input_len)
    position_ids = torch.vstack([torch.arange(input_len) for i in range(input_bs)])
    if model.config.model_type != "opt":
        example_inputs = (input_ids, attention_mask, position_ids, tuple(past_key_values))
    else:
        example_inputs = (input_ids, attention_mask, tuple(past_key_values))
    # do inference to check example_inputs formats
    model(*example_inputs)

    if return_type != "tuple":
        if model.config.model_type != "opt":
            example_inputs = {
                "input_ids": input_ids,
                "past_key_values": tuple(past_key_values),
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
        else:
            example_inputs = {
                "input_ids": input_ids,
                "past_key_values": tuple(past_key_values),
                "attention_mask": attention_mask
            }
    return example_inputs
