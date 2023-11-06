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

# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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
c

import warnings

import torch
import transformers
from intel_extension_for_transformers.transformers import (
    BitsAndBytesConfig,
    MixedPrecisionConfig,
    SmoothQuantConfig,
    WeightOnlyQuantConfig,
)
from intel_extension_for_transformers.transformers.utils.utility import (
    logger,
    LazyImport,
    generate_dummy_past_key_values,
    get_example_inputs_for_trace,
)
from transformers.utils import is_accelerate_available, is_bitsandbytes_available

torch = LazyImport("torch")


class _BaseQBitsAutoModelClass:
    ORIG_MODEL = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import intel_extension_for_transformers.transformers.modeling.modeling_map
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)

        use_llm_runtime = kwargs.pop("use_llm_runtime", True)
        device_map = kwargs.get("device_map", None)
        if isinstance(quantization_config, BitsAndBytesConfig):
            model = cls.ORIG_MODEL.from_pretrained(
                pretrained_model_name_or_path,
                quantization_config=quantization_config,
                *model_args,
                **kwargs,
            )
        if load_in_8bit or load_in_4bit:
            use_cpu = (
                True
                if device_map == torch.device("cpu")
                or device_map == "cpu"
                else False
            )
            if (
                is_accelerate_available()
                and is_bitsandbytes_available()
                and not use_cpu
            ):
                model = cls.ORIG_MODEL.from_pretrained(
                    pretrained_model_name_or_path,
                    quantization_config=quantization_config,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    *model_args,
                    **kwargs,
                )
                logger.info("WeightOnlyQuant bitsandbytes done.")
                return model
            logger.info("CPU device is used.")
            if load_in_8bit or load_in_4bit or quantization_config is not None:
                from intel_extension_for_transformers.llm.quantization.utils import (
                    convert_to_quantized_model,
                )

                torch_dtype = kwargs.pop("torch_dtype", torch.float32)
            if load_in_4bit:
                if quantization_config is None:
                    quantization_config = WeightOnlyQuantConfig(
                        compute_dtype=torch_dtype, weight_dtype="nf4"
                    )
                else:
                    assert (
                        "4" in quantization_config.weight_dtype
                        and quantization_config.compute_dtype == torch_dtype
                    ), "Quantization_config.weight_dtype should be 'nf4', 'int4_fullrange', 'int4_clip',"
                    f"'fp4_e2m1' or 'fp4_e2m1_bnb' and compute_dtype should be {torch_dtype}."
            elif load_in_8bit:
                if quantization_config is None:
                    quantization_config = WeightOnlyQuantConfig(
                        compute_dtype=torch_dtype, weight_dtype="int8"
                    )
                else:
                    assert (
                        quantization_config.weight_dtype == "int8"
                        and quantization_config.compute_dtype == torch_dtype
                    ), f"Quantization_config.weight_dtype should be 'int8' and compute_dtype should be {torch_dtype}."

        if isinstance(quantization_config, MixedPrecisionConfig):
            kwargs["torch_dtype"] = torch.bfloat16
            logger.info("Mixed Precision done.")
        model = cls.ORIG_MODEL.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.eval()
        if isinstance(quantization_config, WeightOnlyQuantConfig):
            logger.info("Applying Weight Only Quantization.")
            if use_llm_runtime:
                logger.info("Using LLM runtime.")
                quantization_config.post_init_runtime()
                from intel_extension_for_transformers.llm.runtime.graph import Model

                model = Model()
                model.init(
                    pretrained_model_name_or_path,
                    weight_dtype=quantization_config.weight_dtype,
                    alg=quantization_config.scheme,
                    group_size=quantization_config.group_size,
                    scale_dtype=quantization_config.scale_dtype,
                    compute_dtype=quantization_config.compute_dtype,
                    use_ggml=quantization_config.use_ggml,
                )
                return model
            else:
                quantization_config.post_init()
                from intel_extension_for_transformers.llm.quantization.utils import (
                    convert_to_quantized_model,
                )

                model = convert_to_quantized_model(model, quantization_config)
            logger.info("WeightOnlyQuant done.")
        elif isinstance(quantization_config, SmoothQuantConfig):
            logger.info("Applying SmoothQuant.")
            try:
                import intel_extension_for_pytorch as ipex
            except ImportError:
                warnings.warn(
                    "Please install Intel Extension for PyTorch to accelerate the model inference."
                )
            assert ipex.__version__ >= "2.1.0+cpu", "Please use Intel Extension for PyTorch >=2.1.0+cpu."
            calib_func = quantization_config.calib_func
            if calib_func is None:
                if quantization_config.tokenizer is None:
                    logger.error(
                        "Please provide the tokenizer or provide calib_func directly,"
                        + " the following is how to get tokenizer. \n"
                        + " from transformer import AutoTokenizer \n"
                        + " tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) \n"
                    )
                    exit(0)
                from datasets import load_dataset
                from torch.utils.data import DataLoader

                calib_dataset = quantization_config.calib_dataset
                calib_iters = quantization_config.calib_iters
                calib_dataset = load_dataset(calib_dataset, split="train")
                calib_dataset = calib_dataset.shuffle(seed=42)

                def tokenize_function(examples):
                    if "prompt" in examples:
                        example = quantization_config.tokenizer(examples["prompt"])
                    elif "text" in examples:
                        example = quantization_config.tokenizer(examples["text"])
                    elif "code" in examples:
                        example = quantization_config.tokenizer(examples["code"])
                    else:
                        logger.error(
                            "Please check dataset prompt identifier,"
                            + " NeelNanda/pile-10k is default used calibration dataset."
                        )
                        exit(0)
                    return example

                tokenized_dataset = calib_dataset.map(tokenize_function, batched=True)
                tokenized_dataset.set_format(type="torch", columns=["input_ids"])

                def collate_batch(batch):
                    input_ids_padded = []
                    for text in batch:
                        input_ids = text["input_ids"]
                        input_ids = (
                            input_ids[:512] if len(input_ids) > 512 else input_ids
                        )
                        input_ids_padded.append(input_ids)
                    return torch.vstack(input_ids_padded)

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
                    past_key_values = generate_dummy_past_key_values(input_bs, model)
                    attention_mask = torch.ones(input_bs, input_len + 1)
                    attention_mask[:, 0] = 0
                    if i >= calib_iters:
                        break
                    model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                    )

            recipes = {
                "smooth_quant": True,
                "smooth_quant_args": {"alpha": quantization_config.alpha},
            }
            example_inputs = get_example_inputs_for_trace(model)
            from neural_compressor import PostTrainingQuantConfig, quantization

            conf = PostTrainingQuantConfig(
                backend="ipex",
                excluded_precisions=quantization_config.excluded_precisions,
                op_type_dict=quantization_config.op_type_dict,
                op_name_dict=quantization_config.op_name_dict,
                recipes=recipes,
                example_inputs=example_inputs,
            )
            if calib_func is None:
                logger.info(
                    "The default calibration funcation is used, "
                    + "the calibration dataset is NeelNanda/pile-10k,"
                    + "batchsize is 1 and calibration iteration is 100."
                )
                calib_func = default_calib_func
            else:
                calib_func = calib_func
            model.config.torchscript = True
            model = quantization.fit(
                                    model, 
                                    conf,
                                    calib_func=calib_func,
                                    calib_dataloader=calib_dataloader if quantization_config.alpha=="auto" else None
                                    )
            logger.info("SmoothQuant done.")
        return model


class AutoModelForCausalLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForCausalLM


class AutoModel(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModel


class AutoModelForSeq2SeqLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForSeq2SeqLM

class GPTBigCodeForCausalLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.GPTBigCodeForCausalLM
