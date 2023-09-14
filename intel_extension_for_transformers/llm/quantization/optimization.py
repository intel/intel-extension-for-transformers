# !/usr/bin/env python
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

from doctest import Example
from typing import Union
from venv import logger
from intel_extension_for_transformers.transformers.utils.utility import LazyImport
from intel_extension_for_transformers.neural_chat.config import (
    AMPConfig,
    WeightOnlyQuantizationConfig,
    BitsAndBytesConfig,
    SmoothQuantConfig
)
import logging
logger = logging.getLogger(__name__)
torch = LazyImport("torch")

class Optimization:
    def __init__(
            self,
            optimization_config: Union[AMPConfig, WeightOnlyQuantizationConfig, BitsAndBytesConfig, SmoothQuantConfig]
        ):
        self.optimization_config = optimization_config

    def optimize(self, model, tokenizer=None, calib_func=None):
        """
        Optimize the model with a given config.
        """
        optimized_model = model
        config = self.optimization_config
        if isinstance(config, WeightOnlyQuantizationConfig):
            print("Applying Weight Only Quantization.")
            from neural_compressor import PostTrainingQuantConfig, quantization
            op_type_dict = {
                '.*':{ 	# re.match
                    "weight": {
                        'bits': config.bits, # 1-8 bits
                        'group_size': config.group_size,  # -1 (per-channel)
                        'scheme': config.scheme, # sym/asym
                        'algorithm': config.algorithm, # RTN/AWQ/TEQ
                    },
                },
            }
            recipes = {"rtn_args": {"enable_full_range": config.enable_full_range}}
            conf = PostTrainingQuantConfig(
                approach='weight_only',
                op_type_dict=op_type_dict,
                recipes=recipes,
            )
            optimized_model = quantization.fit(
                model,
                conf,
            ).model
        elif isinstance(config, SmoothQuantConfig):
            print("Applying SmoothQuant.")
            if tokenizer is None:
                logger.error("Please provide the tokenizer. \n" +
                                "from transformer import AutoTokenizer \n" +
                                "tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) \n" +
                                "Or provide calib_func directly."
                                )
            if calib_func is None:
                from datasets import load_dataset
                from torch.utils.data import DataLoader
                calib_dataset = load_dataset("NeelNanda/pile-10k", split="train")
                calib_dataset = calib_dataset.shuffle(seed=42)

                def tokenize_function(examples):
                    return tokenizer(examples["text"])

                tokenized_dataset = calib_dataset.map(tokenize_function, batched=True)
                tokenized_dataset.set_format(type="torch", columns=["input_ids"])

                def collate_batch(batch):
                    input_ids_padded = []
                    for text in batch:
                        input_ids = text["input_ids"]
                        input_ids = (
                                input_ids[: 512]
                                if len(input_ids) > 512
                                else input_ids
                            )
                        input_ids_padded.append(input_ids)
                    return (torch.vstack(input_ids_padded))
                calib_dataloader = DataLoader(
                    tokenized_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=collate_batch,
                )
            def default_calib_func(model):
                """
                This is the default calibration function, the dataset is NeelNanda/pile-10k,
                the default calib_iters is 100.
                """

                for i, (input_ids) in enumerate(calib_dataloader):
                    input_bs, input_len = input_ids.shape
                    past_key_values = self.generate_dummy_past_key_values(input_bs, model)
                    attention_mask = torch.ones(input_bs, input_len + 1)
                    attention_mask[:,0] = 0
                    if i >= 100:
                        break
                    model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                    )
            recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": config.alpha}}
            example_inputs = self.get_example_inputs_for_trace(model)
            from neural_compressor import PostTrainingQuantConfig, quantization
            conf = PostTrainingQuantConfig(
                backend="ipex",
                excluded_precisions=config.excluded_precisions,
                op_type_dict=config.op_type_dict,
                recipes=recipes,
                example_inputs=example_inputs,
            )
            if calib_func is None:
                logger.info("The default calibration funcation is used, " +
                            "the calibration dataset is NeelNanda/pile-10k," +
                            "batchsize is 1 and calibration iteration is 100.")
                calib_func = default_calib_func
            else:
                calib_func = calib_func
            optimized_model = quantization.fit(
                model,
                conf,
                calib_func=calib_func,
            )
        return optimized_model

    def generate_dummy_past_key_values(self, input_bs, model):
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
        else:
            new_shape = [input_bs, num_attention_heads, 1, d_k]
            dummy_tensor = torch.ones(size=new_shape)
            pkv = tuple(dummy_tensor for _ in range(nb_pkv))
        past_key_values = tuple(tuple(pkv) for _ in range(num_layers))
        return past_key_values

    def get_example_inputs_for_trace(self, model, return_type="tuple"):
        """
            Generate the example_input for tracing, support models load from AutoModelForCausalLM.

        """
        input_ids = model.dummy_inputs["input_ids"]
        input_bs, input_len = input_ids.shape
        past_key_values = self.generate_dummy_past_key_values(input_bs, model)
        attention_mask = torch.ones(input_bs, input_len + 1)
        attention_mask[:,0] = 0
        example_inputs = (input_ids, tuple(past_key_values), attention_mask)
        # do inference to check example_inputs formats
        model(*example_inputs)
        if return_type != "tuple":
            example_inputs = {
                "input_ids": input_ids,
                "past_key_values": tuple(past_key_values),
                "attention_mask": attention_mask
            }
        return example_inputs

