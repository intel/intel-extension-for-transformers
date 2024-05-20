#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
from ...utils import (
    logger,
    LazyImport,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.functional import pad

torch = LazyImport("torch")
IPEX_OPT_LLM_SUPPORTED_DICT = {"2.2": ["gptj", "opt", "llama", "falcon", "chatglm", "baichuan", "gpt-neox"],
                          "2.3": ["gptj", "opt", "llama", "falcon", "chatglm", "baichuan", "bloom", "codegen", "gptbigcode", "t5", "mixtral", "mpt"]}

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

def generate_dummy_past_key_values_for_opt_llm(config, input_bs, num_beams=1):
    """Generate the dummy past_key_values."""
    from optimum.utils import NormalizedConfigManager
    if config.model_type == "qwen":
        new_shape = [input_bs, 1, config.num_attention_heads, normalized_config.hidden_size//config.num_attention_heads]
    elif config.model_type == "baichuan":
        new_shape = [input_bs, config.num_attention_heads, 1, normalized_config.hidden_size//config.num_attention_heads]
    elif config.model_type == "chatglm":
        new_shape = [1, input_bs, config.num_attention_heads, normalized_config.hidden_size//config.num_attention_heads]
    else:
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

def generate_dummy_past_key_values(config, input_bs):
    """Generate the dummy past_key_values."""
    from optimum.utils import NormalizedConfigManager
    if config.model_type == "qwen":
        new_shape = [input_bs, 1, config.num_attention_heads, normalized_config.hidden_size//config.num_attention_heads]
    elif config.model_type == "baichuan":
        new_shape = [input_bs, config.num_attention_heads, 1, normalized_config.hidden_size//config.num_attention_heads]
    elif config.model_type == "chatglm":
        new_shape = [1, input_bs, config.num_attention_heads, normalized_config.hidden_size//config.num_attention_heads]
    else:
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

def get_dataloader(model_type, quantization_config, past_key_values, shuffle=False, padding=False, max_input_length=512, pad_val=None):
    calib_dataset = load_dataset(
    quantization_config.dataset,
    split=(
        "test"
        if quantization_config.dataset in ["mbpp", "openai_humaneval"]
        else "train"
    ),
)
    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=42)

    def tokenize_function(examples):
        if "code" in examples:
            example = quantization_config.tokenizer(examples["code"])
        elif "prompt" in examples:
            example = quantization_config.tokenizer(examples["prompt"])
        elif "text" in examples:
            example = quantization_config.tokenizer(examples["text"])
        else:
            logger.error(
                "Please check dataset prompt identifier,"
                + " NeelNanda/pile-10k is default used calibration dataset."
            )
            exit(0)
        return example

    def collate_batch(batch):
        position_ids_padded = []
        input_ids_padded = []
        last_ind = []
        attention_mask_padded = []
        for text in batch:
            input_ids = text["input_ids"]
            if not padding:
                input_ids = (
                    input_ids[: int(max_input_length)]
                    if len(input_ids) > int(max_input_length)
                    else input_ids
                )  # no_padding
            else:
                pad_len = max_input_length - input_ids.shape[0]
                input_ids = pad(
                    input_ids, (0, pad_len), value=max_input_length
                )

            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
            position_ids_padded.append(position_ids)
        if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
            return (
                {
                    "input_ids": torch.vstack(input_ids_padded),
                    "attention_mask": torch.vstack(attention_mask_padded),
                    "position_ids": torch.vstack(position_ids_padded),
                    "past_key_values": past_key_values,
                },
                torch.tensor(last_ind),
            )
        else:
            return (
                {
                    "input_ids": torch.vstack(input_ids_padded),
                    "attention_mask": torch.vstack(attention_mask_padded),
                    "past_key_values": past_key_values,
                },
                torch.tensor(last_ind),
            )

    tokenized_dataset = calib_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    calib_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return calib_dataloader