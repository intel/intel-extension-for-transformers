#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
import os

import torch
from transformers import AutoConfig
from transformers.file_utils import cached_path, hf_bucket_url

from neural_compressor.utils import logger
from neural_compressor.utils.pytorch import load_file

from .config import CONFIG_NAME, WEIGHTS_NAME, QUANTIZED_WEIGHTS_NAME


class NLPOptimizedModel:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated using the"
            f"`{self.__class__.__name__}.from_pretrained(model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs
    ) -> torch.nn.Module:
        """
        Instantiate a quantized pytorch model from a given Intel Neural Compressor (INC) configuration file.
        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(:obj:`str`, `optional`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
        Returns:
            q_model: Quantized model.
        """
        download_kwarg_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None),
        ]
        download_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in download_kwarg_default}

        config = AutoConfig.from_pretrained(model_name_or_path)
        model_class = eval(f'transformers.{config.architectures[0]}')
        if not QUANTIZED_WEIGHTS_NAME in os.listdir(model_name_or_path):
            logger.info("the prunging/distillation optimized model is loading.")
            model = model_class.from_pretrained(model_name_or_path)
            return model
        else:
            logger.info("the quantization optimized model is loading.")
            keys_to_ignore_on_load_unexpected = copy.deepcopy(
                getattr(model_class, "_keys_to_ignore_on_load_unexpected", None)
            )
            keys_to_ignore_on_load_missing = copy.deepcopy(getattr(model_class, "_keys_to_ignore_on_load_missing", None))

            # Avoid unnecessary warnings resulting from quantized model initialization
            quantized_keys_to_ignore_on_load = [r"zero_point", r"scale", r"packed_params", r"constant"]
            if keys_to_ignore_on_load_unexpected is None:
                model_class._keys_to_ignore_on_load_unexpected = quantized_keys_to_ignore_on_load
            else:
                model_class._keys_to_ignore_on_load_unexpected.extend(quantized_keys_to_ignore_on_load)
            missing_keys_to_ignore_on_load = [r"weight", r"bias"]
            if keys_to_ignore_on_load_missing is None:
                model_class._keys_to_ignore_on_load_missing = missing_keys_to_ignore_on_load
            else:
                model_class._keys_to_ignore_on_load_missing.extend(missing_keys_to_ignore_on_load)

            model = model_class.from_pretrained(model_name_or_path, **kwargs)

            model_class._keys_to_ignore_on_load_unexpected = keys_to_ignore_on_load_unexpected
            model_class._keys_to_ignore_on_load_missing = keys_to_ignore_on_load_missing

            if not os.path.isdir(model_name_or_path) and not os.path.isfile(model_name_or_path):
                config_file = hf_bucket_url(model_name_or_path, filename="best_configure.yaml", revision=download_kwargs["revision]"])

                try:
                    resolved_config_file = cached_path(
                        config_file,
                        cache_dir=download_kwargs["cache_dir"],
                        force_download=download_kwargs["force_download"],
                        resume_download=download_kwargs["resume_download"],
                    )
                except EnvironmentError as err:
                    logger.error(err)
                    msg = (
                        f"Can't load config for '{model_name_or_path}'. Make sure that:\n\n"
                        f"-'{model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                        f"-or '{model_name_or_path}' is a correct path to a directory containing a best_configure.yaml file\n\n"
                    )

                    if download_kwargs["revision]"] is not None:
                        msg += (
                            f"- or {download_kwargs['revision']} is a valid git identifier (branch name, a tag name, or a commit id) that "
                            f"exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"
                        )

                    raise EnvironmentError(msg)

                config_file = hf_bucket_url(model_name_or_path, filename="best_model_weights.pt", revision=download_kwargs["revision]"])
                try:
                    resolved_config_file = cached_path(
                        config_file,
                        cache_dir=download_kwargs["cache_dir"],
                        force_download=download_kwargs["force_download"],
                        resume_download=download_kwargs["resume_download"],
                    )
                except EnvironmentError as err:
                    logger.error(err)
                    msg = (
                        f"Can't load config for '{model_name_or_path}'. Make sure that:\n\n"
                        f"-'{model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                        f"-or '{model_name_or_path}' is a correct path to a directory containing a best_model_weights.pt file\n\n"
                    )

                    if download_kwargs["revision]"] is not None:
                        msg += (
                            f"- or {download_kwargs['revision']} is a valid git identifier (branch name, a tag name, or a commit id) that "
                            f"exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"
                        )

                    raise EnvironmentError(msg)

                model_name_or_path = os.path.dirname(resolved_config_file)

            tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)),
                                        CONFIG_NAME)
            weights_file = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)),
                                        WEIGHTS_NAME)
            assert os.path.exists(
                tune_cfg_file), "tune configure file %s didn't exist" % tune_cfg_file
            assert os.path.exists(
                weights_file), "weight file %s didn't exist" % weights_file
            q_model = load_file(weights_file, tune_cfg_file, model)

            return q_model