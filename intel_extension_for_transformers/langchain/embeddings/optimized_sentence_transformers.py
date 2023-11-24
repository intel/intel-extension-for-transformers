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
import sentence_transformers
from sentence_transformers import SentenceTransformer, models
from intel_extension_for_transformers.transformers import OptimizedModel
import transformers
from transformers import T5Config, MT5Config
from typing import List, Optional

WEIGHTS_NAME = "pytorch_model.bin"

logger = logging.getLogger(__name__)


def _get_best_configure_from_weight(model_name_or_path, **kwargs):
    """Get best configure from weight file."""
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    
    best_configure = {}
    if not os.path.isdir(model_name_or_path) and not os.path.isfile(model_name_or_path):  # pragma: no cover
        # pylint: disable=E0611
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.22.0"):
            from transformers.file_utils import cached_path, hf_bucket_url

            weights_file = hf_bucket_url(model_name_or_path, filename=WEIGHTS_NAME, revision=revision)
            try:
                # Load from URL or cache if already cached
                resolved_weights_file = cached_path(
                    weights_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                )
            except EnvironmentError as err:  # pragma: no cover
                logger.error(err)
                msg = (
                    f"Can't load weights for '{model_name_or_path}'. Make sure that:\n\n"
                    f"- '{model_name_or_path}' is a correct model identifier "
                    f"listed on 'https://huggingface.co/models'\n  (make sure "
                    f"'{model_name_or_path}' is not a path to a local directory with "
                    f"something else, in that case)\n\n- or '{model_name_or_path}' is "
                    f"the correct path to a directory containing a file "
                    f"named one of {WEIGHTS_NAME}\n\n"
                )
                if revision is not None:
                    msg += (
                        f"- or '{revision}' is a valid git identifier "
                        f"(branch name, a tag name, or a commit id) that "
                        f"exists for this model name as listed on its model "
                        f"page on 'https://huggingface.co/models'\n\n"
                    )
                raise EnvironmentError(msg)
        else:
            from pathlib import Path

            from huggingface_hub import hf_hub_download
            from transformers.utils import TRANSFORMERS_CACHE, is_offline_mode

            local_files_only = False
            if is_offline_mode():
                logger.info("Offline mode: forcing local_files_only=True")
                local_files_only = True
            if cache_dir is None:
                cache_dir = TRANSFORMERS_CACHE
            if isinstance(cache_dir, Path):
                cache_dir = str(cache_dir)
            try:
                resolved_weights_file = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{model_name_or_path}'. Make sure that:\n\n"
                    f"- '{model_name_or_path}' is a correct model identifier "
                    f"listed on 'https://huggingface.co/models'\n  (make sure "
                    f"'{model_name_or_path}' is not a path to a local directory with "
                    f"something else, in that case)\n\n- or '{model_name_or_path}' is "
                    f"the correct path to a directory containing a file "
                    f"named one of {WEIGHTS_NAME}\n\n"
                )
                if revision is not None:
                    msg += (
                        f"- or '{revision}' is a valid git identifier "
                        f"(branch name, a tag name, or a commit id) that "
                        f"exists for this model name as listed on its model "
                        f"page on 'https://huggingface.co/models'\n\n"
                    )
                raise EnvironmentError(msg)

        weight = torch.load(resolved_weights_file)
        if "best_configure" in weight:
            best_configure = weight['best_configure']
    else:
        weights_file = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)), WEIGHTS_NAME)
        weight = torch.load(weights_file)
        if "best_configure" in weight:
            best_configure = weight['best_configure']
    return best_configure

class OptimzedTransformer(models.Transformer):
    def __init__(self, *args, **kwargs):
        """Initialize the OptimzedTransformer."""
        super().__init__(*args, **kwargs)

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
        return OptimzedTransformer(model_name_or_path=input_path, **config)
    
    def save(self, output_path: str):
        self.tokenizer.save_pretrained(output_path)
        best_configure = _get_best_configure_from_weight(self.auto_model.config._name_or_path)
        state_dict = self.auto_model.state_dict()
        state_dict['best_configure'] = best_configure
        torch.save(state_dict, os.path.join(output_path, WEIGHTS_NAME))
        # save configure dtype as int8 for load identification
        self.auto_model.config.architectures = [self.auto_model.__class__.__name__]
        self.auto_model.config.torch_dtype = "int8"
        self.auto_model.config.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

class OptimizedSentenceTransformer(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        """Initialize the OptimizedSentenceTransformer."""
        super().__init__(*args, **kwargs)

    def _load_auto_model(self, model_name_or_path):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logger.warning("No sentence-transformers model found with name {}." \
                       "Creating a new one with MEAN pooling.".format(model_name_or_path))
        transformer_model = OptimzedTransformer(model_name_or_path)
        pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), 'mean')
        return [transformer_model, pooling_model]
    
    def save(self, path: str, 
             model_name: Optional[str] = None, 
             create_model_card: bool = True, 
             train_datasets: Optional[List[str]] = None):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        :param path: Path on disc
        :param model_name: Optional model name
        :param create_model_card: If True, create a README.md with basic information about this model
        :param train_datasets: Optional list with the names of the datasets used to to train the model
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        logger.info("Save model to {}".format(path))
        modules_config = []

        #Save some model info
        if '__version__' not in self._model_config:
            self._model_config['__version__'] = {
                    'sentence_transformers': sentence_transformers.__version__,
                    'transformers': transformers.__version__,
                    'pytorch': torch.__version__,
                }

        with open(os.path.join(path, 'config_sentence_transformers.json'), 'w') as fOut:
            json.dump(self._model_config, fOut, indent=2)

        #Save modules
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            #Save transformer model in the main folder
            if idx == 0 and isinstance(module, models.Transformer):    
                model_path = path + "/"
            else:
                model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            modules_config.append({'idx': idx, 
                                   'name': name, 
                                   'path': os.path.basename(model_path), 
                                   'type': module.__class__.__module__ + '.' + module.__class__.__name__ \
                                           if isinstance(module, models.Transformer) else \
                                           type(module).__module__})
            
        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(modules_config, fOut, indent=2)

        # Create model card
        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)