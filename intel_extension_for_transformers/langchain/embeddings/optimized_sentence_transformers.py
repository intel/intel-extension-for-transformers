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
from collections import OrderedDict
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
            if isinstance(self.auto_model, torch.jit.ScriptModule):
                setattr(self.auto_model, "config", config)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        if isinstance(self.auto_model, torch.jit.ScriptModule):
            output_states = self.auto_model(**trans_features)
            if isinstance(output_states, dict):
                output_states = tuple(output_states.values())
            output_tokens = output_states[0]
        else:
            output_states = self.auto_model(**trans_features, return_dict=False)
            output_tokens = output_states[0]

        features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return features

class OptimizedSentenceTransformer(sentence_transformers.SentenceTransformer):
    def __init__(self, *args, **kwargs):
        """Initialize the OptimizedSentenceTransformer."""
        self._jit_model = False
        super().__init__(*args, **kwargs)

    def _load_auto_model(
            self,
            model_name_or_path: str,
            token: Optional[Union[bool, str]],
            cache_folder: Optional[str],
            revision: Optional[str] = None,
            trust_remote_code: bool = False):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logger.warning("No sentence-transformers model found with name {}." \
                       "Creating a new one with MEAN pooling.".format(model_name_or_path))
        transformer_model = OptimzedTransformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            )
        if isinstance(transformer_model.auto_model, torch.jit.ScriptModule):
            self._jit_model = True
        pooling_model = sentence_transformers.models.Pooling(
            transformer_model.get_word_embedding_dimension(), 'mean')
        return [transformer_model, pooling_model]

    def encode(self, sentences, device=None, *args, **kwargs):
        if self._jit_model and device is None:
            # set default device to 'cpu' for jit model, otherwise will fail when getting device
            return super().encode(sentences, device='cpu', *args, **kwargs)
        else:
            return super().encode(sentences, device=device, *args, **kwargs)
