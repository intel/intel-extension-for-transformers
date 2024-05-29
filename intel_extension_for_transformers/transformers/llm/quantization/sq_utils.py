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
import re
from typing import Optional, Tuple

import transformers
from datasets import load_dataset
from optimum.intel.generation.modeling import TSModelForCausalLM
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast

from intel_extension_for_transformers.tools.utils import is_ipex_available

from ...utils import LazyImport, logger

if is_ipex_available():
    import intel_extension_for_pytorch as ipex
torch = LazyImport("torch")

IPEX_OPT_LLM_SUPPORTED_DICT = {
    "2.2": ["gptj", "opt", "llama", "falcon", "chatglm", "baichuan", "gpt-neox"],
    "2.3": [
        "gptj",
        "opt",
        "llama",
        "falcon",
        "chatglm",
        "baichuan",
        "qwen",
        "bloom",
        "codegen",
        "gptbigcode",
        "t5",
        "mixtral",
        "mpt",
    ],
}
if is_ipex_available() and ipex.__version__ == "2.2.0+cpu":
    logger.info(
        "ipex.llm.optimize by 2.2.0 version supported model family: {}".format(
            ",".join(IPEX_OPT_LLM_SUPPORTED_DICT["2.2"])
            )
    )
    logger.info(
        "The recommended transformers version is 4.35.2 if you used IPEX 2.2.0 version."
    )
    IPEX_OPT_LLM_SUPPORTED = IPEX_OPT_LLM_SUPPORTED_DICT["2.2"]
elif is_ipex_available() and ipex.__version__ == "2.3.0+cpu":
    logger.info(
        "ipex.llm.optimize by 2.3.0 version supported model family: {}".format(
            ", ".join(IPEX_OPT_LLM_SUPPORTED_DICT["2.3"])
        )
    )
    logger.info(
        "The recommended transformers version is 4.38.1 if you used IPEX 2.3.0 version."
    )
    IPEX_OPT_LLM_SUPPORTED = IPEX_OPT_LLM_SUPPORTED_DICT["2.3"]
else:
    logger.warning("Please check the intel_extension_for_pytorch version is 2.3.0+cpu.")
    IPEX_OPT_LLM_SUPPORTED = IPEX_OPT_LLM_SUPPORTED_DICT["2.3"]

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
    "baichuan",
}


def generate_dummy_past_key_values_for_opt_llm(config, input_bs, num_beams=1):
    """Generate the dummy past_key_values."""
    from optimum.utils import NormalizedConfigManager

    if config.model_type == "qwen":
        new_shape = [
            input_bs,
            1,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        ]
        num_layers = config.num_hidden_layers
    elif config.model_type == "baichuan":
        new_shape = [
            input_bs,
            config.num_attention_heads,
            1,
            config.hidden_size // config.num_attention_heads,
        ]
        num_layers = config.num_hidden_layers
    elif config.model_type == "chatglm":
        new_shape = [
            1,
            input_bs,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        ]
        num_layers = config.num_layers
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
        new_shape = [
            input_bs,
            1,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        ]
        num_layers = config.num_hidden_layers
    elif config.model_type == "baichuan":
        new_shape = [
            input_bs,
            config.num_attention_heads,
            1,
            config.hidden_size // config.num_attention_heads,
        ]
        num_layers = config.num_hidden_layers
    elif config.model_type == "chatglm":
        new_shape = [
            1,
            input_bs,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        ]
        num_layers = config.num_layers
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


def get_dataloader(
    model_type,
    quantization_config,
    past_key_values,
    shuffle=False,
    padding=False,
    seq_len=512,
):
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
                    input_ids[: int(seq_len)]
                    if len(input_ids) > int(seq_len)
                    else input_ids
                )  # no_padding
            else:
                pad_len = seq_len - input_ids.shape[0]
                input_ids = pad(input_ids, (0, pad_len), value=seq_len)

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


class TSModelCausalLMForITREX(TSModelForCausalLM):
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
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
        if len(past_key_values[0]) == 4:  # discrete kv_cache
            for layer_past in past_key_values:
                layer_past[3][layer_past[0].size(-2) - 1] = beam_idx
            return past_key_values
        else:
            return tuple(
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                )
                for layer_past in past_key_values
            )

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        past_key_values = past_key_values or kwargs.get("past", None)

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # `past_key_values` may be in the standard format (e.g. in contrastive search),
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
        input_bs, input_len = input_ids.shape

        if self.use_cache and past_key_values is None:
            if model_type in IPEX_OPT_LLM_SUPPORTED:
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

def loading_configure_file(model, json_file_path, example_inputs):
    """Recover ipex model from JSON file.

    Args:
        model (object): fp32 model need to do quantization.
        json_file_path (json): configuration JSON file for ipex.
        example_inputs (tuple or torch.Tensor or dict): example inputs that will be passed to the ipex function.

    Returns:
        (object): quantized model
    """

    ipex = LazyImport("intel_extension_for_pytorch")
    from torch.ao.quantization.observer import MinMaxObserver

    if ipex.__version__ >= "2.1.100":
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5, act_observer=MinMaxObserver)
    else:
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5, act_observer=MinMaxObserver())
    if isinstance(example_inputs, dict):
        model = ipex.quantization.prepare(model, qconfig, example_kwarg_inputs=example_inputs, inplace=True)
    else:
        model = ipex.quantization.prepare(model, qconfig, example_inputs=example_inputs, inplace=True)
    model.load_qconf_summary(qconf_summary=json_file_path)
    model = ipex.quantization.convert(model, inplace=True)
    model.eval()
    with torch.no_grad():
        model = torch.jit.trace(model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False)
        model = torch.jit.freeze(model.eval())

    model(**example_inputs)
    model(**example_inputs)
    return model

def recover_model_from_json(fp32_model_name_or_path, json_file_path, trust_remote_code=False):
    """Recover ipex model from JSON file.

    Args:
        model (object): fp32 model need to do quantization.
        json_file_path (json): configuration JSON file for ipex saved.
        trust_remote_code (bool): trust remote code.

    Returns:
        (object): quantized model
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(fp32_model_name_or_path, trust_remote_code=trust_remote_code)
    if model.config.model_type in IPEX_OPT_LLM_SUPPORTED:
        qconfig = ipex.quantization.default_static_qconfig_mapping
        model = ipex.llm.optimize(
            model.eval(),
            dtype=torch.float,
            inplace=True,
            quantization_config=qconfig,
            deployment_mode=False,
        )
    # config
    model.config.torchscript = True
    config = model.config

    # example_inputs

    input_ids= model.dummy_inputs["input_ids"]
    input_bs, input_len = input_ids.shape
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_len).repeat(input_bs, 1)
    num_beams = 1
    if config.model_type in IPEX_OPT_LLM_SUPPORTED:
        past_key_values = generate_dummy_past_key_values_for_opt_llm(
            config=config, input_bs=input_bs, num_beams=num_beams
        )
    else:
        past_key_values = generate_dummy_past_key_values(
            config=config, input_bs=input_bs
        )
    if config.model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
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

    model = loading_configure_file(model, json_file_path, example_inputs)
    model = TSModelCausalLMForITREX(model, config=config)
    return model
