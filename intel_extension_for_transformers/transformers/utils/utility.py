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

import argparse
import os
from typing import Optional, Tuple
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport


CONFIG_NAME = "best_configure.yaml"
ENGINE_MODEL_NAME = "model.bin"
ENGINE_MODEL_CONFIG = "conf.yaml"
ENCODER_NAME = "encoder_model.bin"
DECODER_NAME = "decoder_model.bin"
DECODER_WITH_PAST_NAME = "decoder_with_past_model.bin"
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
QUANT_CONFIG = "quantization_config.json"
SPARSITY_CONFIG = "sparsity_config.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

torch = LazyImport("torch")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def distributed_init(
    backend="gloo",
    world_size=1,
    rank=-1,
    init_method=None,
    master_addr="127.0.0.1",
    master_port="12345",
):
    """Init the distribute environment."""
    rank = int(os.environ.get("RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", world_size))
    if init_method is None:
        master_addr = os.environ.get("MASTER_ADDR", master_addr)
        master_port = os.environ.get("MASTER_PORT", master_port)
        init_method = "env://{addr}:{port}".format(addr=master_addr, port=master_port)
    torch.distributed.init_process_group(
        backend, init_method=init_method, world_size=world_size, rank=rank
    )


def remove_label(input):
    if "labels" in input:  # for GLUE
        input.pop("labels")
    elif "start_positions" in input and "end_positions" in input:  # for SQuAD
        # pragma: no cover
        input.pop("start_positions")
        input.pop("end_positions")
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
        shape_key = (input_bs * num_attention_heads, d_k, 1)
        shape_value = (input_bs * num_attention_heads, 1, d_k)
        key = torch.ones(size=shape_key)
        value = torch.ones(size=shape_value)
        past_key_values = tuple(
            tuple(key if idx % 2 == 0 else value for idx in range(nb_pkv))
            for _ in range(num_layers)
        )
        return past_key_values
    elif config.model_type == "gpt_bigcode":
        new_shape = [input_bs, 0, d_k * 2]
        dummy_tensor = torch.zeros(size=new_shape)
        past_key_values = tuple([dummy_tensor] * num_layers)
        return past_key_values
    elif config.model_type == "qwen":
        new_shape = [input_bs, 1, num_key_value_heads, d_k]
        past_key_values = [
            (
                torch.ones(size=new_shape).contiguous(),
                torch.ones(size=new_shape).contiguous(),
            )
            for _ in range(num_layers)
        ]
        return tuple(past_key_values)
    elif config.model_type == "chatglm":
        new_shape = [0, input_bs, num_key_value_heads, d_k]
    elif config.model_type == "falcon":
        new_shape = [input_bs, 1, 0, d_k]
    else:
        new_shape = [input_bs, num_key_value_heads, 0, d_k]
    past_key_values = [
        (
            torch.zeros(size=new_shape).contiguous(),
            torch.zeros(size=new_shape).contiguous(),
        )
        for _ in range(num_layers)
    ]
    return tuple(past_key_values)

def generate_dummy_past_key_values_for_inference(config, input_bs):
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
        shape_key = (input_bs * num_attention_heads, d_k, 0)
        shape_value = (input_bs * num_attention_heads, 0, d_k)
        key = torch.empty(size=shape_key)
        value = torch.empty(size=shape_value)
        past_key_values = tuple(
            tuple(key if idx % 2 == 0 else value for idx in range(nb_pkv))
            for _ in range(num_layers)
        )
        return past_key_values
    elif config.model_type == "gpt_bigcode":
        new_shape = [input_bs, 0, d_k * 2]
        dummy_tensor = torch.zeros(size=new_shape)
        past_key_values = tuple([dummy_tensor] * num_layers)
        return past_key_values
    elif config.model_type == "qwen":
        new_shape = [input_bs, 0, num_key_value_heads, d_k]
    elif config.model_type == "chatglm":
        new_shape = [0, input_bs, num_key_value_heads, d_k]
    elif config.model_type == "falcon":
        new_shape = [input_bs, 1, 0, d_k]
    else:
        new_shape = [input_bs, num_key_value_heads, 0, d_k]
    past_key_values = [
        (
            torch.zeros(size=new_shape).contiguous(),
            torch.zeros(size=new_shape).contiguous(),
        )
        for _ in range(num_layers)
    ]
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
    past_key_values = [
        (
            torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
            torch.zeros(size=new_shape).contiguous(),
            torch.zeros(size=new_shape).contiguous(),
            beam_idx_tmp,
        )
        for _ in range(num_layers)
    ]
    return tuple(past_key_values)


IPEX_OPT_LLM_SUPPORTED = {"gptj", "opt", "llama", "falcon", "chatglm", "baichuan"}

MODEL_TYPES_REQUIRING_POSITION_IDS = {
    "codegen",
    "gpt2",
    "gpt-bigcode",
    "gpt-neo",
    "gpt-neox",
    "gptj",
    "imagegpt",
    "llama",
    "mistral",
    "chatglm",
    "baichuan"
}

def get_example_inputs(model_config, batch_size=1, tokenizer=None, num_beams=4):
    """Generate the dummy example inputs.
    """
    prompt = "Welcome to use Intel Extension for Transformers."
    prompt = [prompt] * batch_size
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if model_config.model_type in IPEX_OPT_LLM_SUPPORTED:
        past_key_values = generate_dummy_past_key_values_for_opt_llm(
                                                                    config=model_config,
                                                                    input_bs=batch_size,
                                                                    num_beams=num_beams
                                                                    )
    else:
        past_key_values = generate_dummy_past_key_values(config=model_config, input_bs=batch_size)

    input_ids = input_ids[:, :512]
    attention_mask = torch.ones(input_ids.shape)
    position_ids = torch.arange(input_ids.shape[1]).repeat(batch_size, 1)

    if model_config.model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
        example_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": past_key_values
                }
    else:
        example_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values
                }
    return example_inputs

def recover_model_from_json(user_model, json_file_path, trust_remote_code=False):
    """Recover ipex model from JSON file.

    Args:
        model (object): fp32 model need to do quantization.
        json_file_path (json): configuration JSON file for ipex.
        trust_remote_code (bool): trust remote code.

    Returns:
        (object): quantized model
    """
    if user_model.config.model_type in IPEX_OPT_LLM_SUPPORTED:
        import intel_extension_for_pytorch as ipex
        qconfig = ipex.quantization.default_static_qconfig_mapping
        user_model = ipex.optimize_transformers(
            user_model.eval(),
            dtype=torch.float,
            inplace=True,
            quantization_config=qconfig,
            deployment_mode=False,
        )

    # tokenizer
    if user_model.config.model_type == "llama":
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(user_model.config.name_or_path)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            user_model.config.name_or_path, trust_remote_code=trust_remote_code
        )

    # example_inputs
    example_inputs = get_example_inputs(user_model.config, tokenizer=tokenizer)

    # pylint: disable=E0611
    from neural_compressor.utils.pytorch import recover_model_from_json as inc_recover_model_from_json
    user_model = inc_recover_model_from_json(user_model, json_file_path, example_inputs)
    return user_model
