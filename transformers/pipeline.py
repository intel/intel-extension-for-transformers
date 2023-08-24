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
"""Pipeline: import transformers.pipelines and support int8 model loading based on infer_framework_load_model."""
import importlib
from transformers import AutoConfig, pipeline
from transformers.pipelines import *
from typing import Dict, Optional, Tuple
from .model import OptimizedModel


origin_forward = Pipeline.forward
origin_check = Pipeline.check_model_type

# pylint: disable=E0102
def infer_framework_load_model(
    model,
    config: AutoConfig,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs
):
    """Support int8 model loading based on infer_framework_load_model.

    Args:
        model (object): the input model
        config (AutoConfig): AutoConfig object
        model_classes (Optional[Dict[str, Tuple[type]]], optional): model class. Defaults to None
        task (Optional[str], optional): task name. Defaults to None
        framework (Optional[str], optional): framework name. Defaults to None

    Returns:
        Tuple: A tuple framework, model.
    """
    logger.warning("Function transformers.pipelines.base.infer_framework_load_model is replaced "
                    "by intel_extension_for_transformers.optimization.pipeline.")

    backend = model_kwargs['backend'] if 'backend' in model_kwargs else None
    if isinstance(model, str):
        if backend == 'executor':  # pragma: no cover
            from intel_extension_for_transformers.backends.neural_engine.compile import compile
            model = compile(model)
            model.__call__= model.inference
            model.config = config
            framework = 'pt'

            # only support text-classification now
            def forward_executor(self, model_inputs, **forward_params):
                """Forward function executor.

                Args:
                    model_inputs (list): model inputs
                """
                model_inputs = [v.int() for k, v in model_inputs.items()]
                model_outputs = model.inference(model_inputs)
                model_outputs = list(self.model.inference(model_inputs).values())[0]
                return {"logits": torch.from_numpy(model_outputs)}

            def _check_model_type(self, supported_models):
                pass

            Pipeline.forward = forward_executor
            Pipeline.check_model_type = _check_model_type
        else:
            model = OptimizedModel.from_pretrained(model, **model_kwargs)
            if hasattr(model, "eval"):
                model.eval()
            framework = "tf" if model.__class__.__name__.startswith("TF") else "pt"

            Pipeline.forward = origin_forward
            Pipeline.check_model_type = origin_check

    return framework, model


# Replace the function in pipeline to support int8 loading
trans_pipeline = importlib.import_module('transformers.pipelines')
trans_pipeline.infer_framework_load_model = infer_framework_load_model
