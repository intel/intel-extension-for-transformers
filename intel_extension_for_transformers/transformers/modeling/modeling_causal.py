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

import copy

from transformers import AutoConfig, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.auto.auto_factory import _get_model_class
from intel_extension_for_transformers.transformers.utils.utility import (
    LazyImport,
    generate_dummy_past_key_values,
    get_example_inputs_for_trace
)


from intel_extension_for_transformers.transformers import (
    AMPConfig,
    WeightOnlyQuantizationConfig,
    SmoothQuantConfig
)
import logging
import warnings
logger = logging.getLogger(__name__)
torch = LazyImport("torch")


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import intel_extension_for_transformers.transformers.modeling.modeling_map
        config = kwargs.pop("config", None)
        calib_func = kwargs.pop("calib_func", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "code_revision",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}

        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype", None) == "auto":
                _ = kwargs.pop("torch_dtype")
            # to not overwrite the quantization_config if config has a quantization_config

            if kwargs.get("quantization_config", None) is not None:
                _ = kwargs.pop("quantization_config")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
                **kwargs,
            )

            # if torch_dtype=auto was passed here, ensure to pass it on
            if kwargs_orig.get("torch_dtype", None) == "auto":
                kwargs["torch_dtype"] = "auto"
            quantization_config = kwargs_orig.get("quantization_config", None)
            if quantization_config is not None and not (isinstance(quantization_config, SmoothQuantConfig) or 
                                                        isinstance(quantization_config, AMPConfig) or
                                                        isinstance(quantization_config, WeightOnlyQuantizationConfig)
                                                        ):
                kwargs["quantization_config"] = kwargs_orig["quantization_config"]
            if isinstance(quantization_config, AMPConfig):
                config.torch_dtype=torch.bfloat16

        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, **hub_kwargs, **kwargs
            )
            _ = hub_kwargs.pop("code_revision", None)
            model = model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            model =  model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        else:
            raise ValueError(
                f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
                f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
            )
        model.eval()
        if isinstance(quantization_config, WeightOnlyQuantizationConfig):
            logger.info("Applying Weight Only Quantization.")
            from neural_compressor import PostTrainingQuantConfig, quantization
            op_type_dict = {
                '.*':{ 	# re.match
                    "weight": {
                        'bits': quantization_config.bits, # 1-8 bits
                        'group_size': quantization_config.group_size,  # -1 (per-channel)
                        'scheme': quantization_config.scheme, # sym/asym
                        'algorithm': quantization_config.algorithm, # RTN/AWQ/TEQ
                    },
                },
            }
            recipes = {"rtn_args": {"enable_full_range": quantization_config.enable_full_range}}
            conf = PostTrainingQuantConfig(
                approach='weight_only',
                op_type_dict=op_type_dict,
                recipes=recipes,
            )
            model.config.torchscript = True
            model = quantization.fit(
                model,
                conf,
            ).model
        elif isinstance(quantization_config, SmoothQuantConfig):
            logger.info("Applying SmoothQuant.")
            try:
                import intel_extension_for_pytorch as ipex
            except ImportError:
                warnings.warn(
                    "Please install Intel Extension for PyTorch to accelerate the model inference."
                )
            if quantization_config.tokenizer is None:
                logger.error("Please provide the tokenizer or provide calib_func directly," + 
                                " the following is how to get tokenizer. \n" +
                                " from transformer import AutoTokenizer \n" +
                                " tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) \n"
                                )
                exit(0)
            if calib_func is None:
                from datasets import load_dataset
                from torch.utils.data import DataLoader
                calib_dataset = quantization_config.calib_dataset
                calib_iters = quantization_config.calib_iters
                calib_dataset = load_dataset(calib_dataset, split="train")
                calib_dataset = calib_dataset.shuffle(seed=42)

                def tokenize_function(examples):
                    if 'prompt' in examples:
                        example = quantization_config.tokenizer(examples["prompt"])
                    elif 'text' in examples:
                        example = quantization_config.tokenizer(examples["text"])
                    elif 'code' in examples:
                        example = quantization_config.tokenizer(examples["code"])
                    else:
                        logger.error("Please check dataset prompt identifier," +
                                     " NeelNanda/pile-10k is default used calibration dataset.")
                        exit(0)
                    return example

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
                    past_key_values = generate_dummy_past_key_values(input_bs, model)
                    attention_mask = torch.ones(input_bs, input_len + 1)
                    attention_mask[:,0] = 0
                    if i >= calib_iters:
                        break
                    model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                    )
            recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": quantization_config.alpha}}
            example_inputs = get_example_inputs_for_trace(model)
            from neural_compressor import PostTrainingQuantConfig, quantization
            conf = PostTrainingQuantConfig(
                backend="ipex",
                excluded_precisions=quantization_config.excluded_precisions,
                op_type_dict=quantization_config.op_type_dict,
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
            model.config.torchscript = True
            model = quantization.fit(
                model,
                conf,
                calib_func=calib_func
            ).model
        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING