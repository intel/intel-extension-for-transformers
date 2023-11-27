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

import logging
from transformers.utils import is_torch_available
from optimum.exporters.tasks import TasksManager
from optimum.exporters.onnx import (
    get_encoder_decoder_models_for_export,
    get_decoder_models_for_export,
    get_stable_diffusion_models_for_export,
)

if is_torch_available():
    import torch

from typing import Any, Callable, Dict, List, Optional, Union
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

def get_onnx_configs(
    model: PreTrainedModel,
    task: str,
    monolith: bool = False,
    custom_onnx_configs: Dict = {},
    _variant: str = "default",
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    preprocessors: Optional[List[Any]] = None,
    legacy: bool = False,
):
    """Get config for onnx export.

    Args:
        model (PreTrainedModel): _description_
        task (str): _description_
        monolith (bool, optional): Forces to export the model as a single ONNX file. Defaults to False.
        custom_onnx_configs (Dict, optional):  override the default ONNX config used for the given model. 
            Defaults to {}.
        _variant (str, optional): Specify the variant of the ONNX export to use. 
            Defaults to "default".
        int_dtype (str, optional): The data type of integer tensors, could be ["int64", "int32", "int8"]. 
            Defaults to "int64".
        float_dtype (str, optional): The data type of float tensors, could be ["fp32", "fp16", "bf16"]. 
            Defaults to "fp32".
        preprocessors (Optional[List[Any]], optional): preprocessors. Defaults to None.
        legacy (bool, optional): Disable the use of position_ids for text-generation models that 
            require it for batched generation. Also enable to export decoder only models in three 
            files (without + with past and the merged model). Defaults to False.

    Returns:
        _type_: _description_
    """
    is_stable_diffusion = "stable-diffusion" in task
    if is_stable_diffusion: # pragma: no cover
        onnx_config = None
        models_and_onnx_configs = get_stable_diffusion_models_for_export(
            model, int_dtype=int_dtype, float_dtype=float_dtype
        )
    else:
        onnx_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model, exporter="onnx", task=task
        )
        onnx_config = onnx_config_constructor(
            model.config,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            legacy=legacy,
        )

        onnx_config.variant = _variant
        all_variants = "\n".join(
            [f"    - {name}: {description}" for name, description in onnx_config.VARIANTS.items()]
        )
        logger.info(f"Using the export variant {onnx_config.variant}. Available variants are:\n{all_variants}")

        if (
            model.config.is_encoder_decoder
            and task.startswith(TasksManager._ENCODER_DECODER_TASKS)
            and not monolith
        ): # pragma: no cover
            models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
        elif task.startswith("text-generation") and not monolith: # pragma: no cover
            models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config, legacy=legacy)
        else:
            models_and_onnx_configs = {"model": (model, onnx_config)}

    # When specifying custom ONNX configs for supported transformers architectures, we do
    # not force to specify a custom ONNX config for each submodel.
    for key, custom_onnx_config in custom_onnx_configs.items():
        models_and_onnx_configs[key] = (models_and_onnx_configs[key][0], custom_onnx_config)

    # Default to the first ONNX config for stable-diffusion and custom architecture case.
    if onnx_config is None:
        onnx_config = next(iter(models_and_onnx_configs.values()))[1]

    return onnx_config
