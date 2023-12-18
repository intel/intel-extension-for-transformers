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

import os
import json
import logging
import torch
from intel_extension_for_transformers.transformers import OptimizedModel
from intel_extension_for_transformers.transformers.utils.utility import LazyImport
import transformers
from transformers import T5Config, MT5Config
from typing import Union, Optional

sentence_transformers = LazyImport("sentence_transformers")

WEIGHTS_NAME = "pytorch_model.bin"

logger = logging.getLogger(__name__)

class OptimzedTransformer(sentence_transformers.models.Transformer):
    def __init__(self, *args, **kwargs):
        """Initialize the OptimzedTransformer."""
        super().__init__(*args, **kwargs)

    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        if isinstance(config, T5Config): # pragma: no cover
            self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config): # pragma: no cover
            self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            self.auto_model = OptimizedModel.from_pretrained(model_name_or_path, 
                                                             config=config, 
                                                             cache_dir=cache_dir, 
                                                             **model_args)

class OptimizedSentenceTransformer(sentence_transformers.SentenceTransformer):
    def __init__(self, *args, **kwargs):
        """Initialize the OptimizedSentenceTransformer."""
        super().__init__(*args, **kwargs)

    def _load_auto_model(self, model_name_or_path: str, token: Optional[Union[bool, str]], cache_folder: Optional[str]):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logger.warning("No sentence-transformers model found with name {}." \
                       "Creating a new one with MEAN pooling.".format(model_name_or_path))
        transformer_model = OptimzedTransformer(model_name_or_path, cache_dir=cache_folder, model_args={"token": token})
        pooling_model = sentence_transformers.models.Pooling(transformer_model.get_word_embedding_dimension(), 'mean')
        return [transformer_model, pooling_model]