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
from collections import OrderedDict
from intel_extension_for_transformers.transformers import OptimizedModel
from intel_extension_for_transformers.transformers.utils.utility import LazyImport
from transformers import T5Config, MT5Config

from .optimized_sentence_transformers import OptimzedTransformer

sentence_transformers = LazyImport("sentence_transformers")
InstructorEmbedding = LazyImport("InstructorEmbedding")

logger = logging.getLogger(__name__)

class OptimizedInstructorTransformer(InstructorEmbedding.INSTRUCTOR_Transformer):
    def __init__(self, *args, **kwargs):
        """Initialize the OptimizedInstructorTransformer."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 
                            'sentence_distilbert_config.json', 'sentence_camembert_config.json', 
                            'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 
                            'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return OptimizedInstructorTransformer(model_name_or_path=input_path, **config)
    
    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config):
            self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            self.auto_model = OptimizedModel.from_pretrained(model_name_or_path, 
                                                             config=config, 
                                                             cache_dir=cache_dir, 
                                                             **model_args)

class OptimizedInstructor(InstructorEmbedding.INSTRUCTOR):
    def __init__(self, *args, **kwargs):
        """Initialize the OptimizedInstructor."""
        super().__init__(*args, **kwargs)

    def _load_auto_model(self, model_name_or_path): # pragma: no cover
        """Creates a simple Transformer + Mean Pooling model and returns the modules."""
        logger.warning("No sentence-transformers model found with name {}." \
                       "Creating a new one with MEAN pooling.".format(model_name_or_path))
        transformer_model = OptimzedTransformer(model_name_or_path)
        pooling_model = sentence_transformers.models.Pooling(transformer_model.get_word_embedding_dimension(), 'mean')
        return [transformer_model, pooling_model]
    
    def _load_sbert_model(self, model_path):
        """Loads a full sentence-transformers model."""
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = os.path.join(model_path, 'config_sentence_transformers.json')
        if os.path.exists(config_sentence_transformers_json_path):
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)

        # Check if a readme exists
        model_card_path = os.path.join(model_path, 'README.md')
        if os.path.exists(model_card_path):
            try:
                with open(model_card_path, encoding='utf8') as fIn:
                    self._model_card_text = fIn.read()
            except:
                pass

        # Load the modules of sentence transformer
        modules_json_path = os.path.join(model_path, 'modules.json')
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)

        modules = OrderedDict()
        for module_config in modules_config:
            if module_config['idx']==0:
                logger.info('load Optimized InstructorTransformer')
                module_class = OptimizedInstructorTransformer
            elif module_config['idx']==1:
                module_class = InstructorEmbedding.INSTRUCTOR_Pooling
            else:
                module_class = InstructorEmbedding.import_from_string(module_config['type'])
            module = module_class.load(os.path.join(model_path, module_config['path']))
            modules[module_config['name']] = module

        return modules