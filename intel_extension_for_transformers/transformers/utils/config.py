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
"""Configs for intel extension for transformers."""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from .utility import QUANT_CONFIG, SPARSITY_CONFIG, LazyImport, logger
import transformers
from transformers import BitsAndBytesConfig, PretrainedConfig

torch = LazyImport("torch")


@dataclass
class MixedPrecisionConfig:
    dtype: str = "bfloat16"


@dataclass
class SmoothQuantConfig:
    backend: str = "ipex"
    ipex_opt_llm: bool = None
    tokenizer: Any = None
    calib_func: Any = None
    calib_dataset: str = "NeelNanda/pile-10k"
    calib_shuffle: bool = True
    calib_iters: int = 100
    calib_padding: bool = False
    calib_len: int = 512
    calib_pad_val: int = 1
    alpha: float = 0.5
    op_type_dict: dict = None
    op_name_dict: dict = None
    excluded_precisions: list = field(default_factory=list)
    example_inputs: Any = None
    num_beams: int = 1
    recipes: dict = field(
        default_factory=lambda: {
            "smooth_quant": True,
            "smooth_quant_args": {"alpha": 0.5},
        }
    )


class SparsityConfig(PretrainedConfig):
    def __init__(
        self,
        sparse_pattern: str = "1x1",
        sparse_dtype=None,
        sparse_layers=None,
        dense_layers: list = ["lm_head"],
        group_size=None,
        **kwargs,
    ):
        self.sparse_pattern = sparse_pattern
        self.sparse_dtype = sparse_dtype
        self.sparse_layers = sparse_layers
        self.dense_layers = dense_layers
        self.group_size = group_size

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        pass

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`SparsityConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`SparsityConfig`]: The configuration object instantiated from those parameters.
        """

        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file_path, return_unused_kwargs, **kwargs):
        with open(json_file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict, return_unused_kwargs, **kwargs)

    def to_json_file(
        self, json_file_path: Union[str, os.PathLike], use_diff: bool = True
    ):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """

        output = copy.deepcopy(self.__dict__)
        return output

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `SparsityConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = SparsityConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, SPARSITY_CONFIG)

        self.to_json_file(output_config_file, use_diff=False)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token", None),
            )

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return super().get_config_dict(
            pretrained_model_name_or_path, _configuration_file=SPARSITY_CONFIG, **kwargs
        )

if transformers.__version__ >= "4.32.0":
    from transformers.utils.quantization_config import QuantizationConfigMixin
    QuantizationConfig = QuantizationConfigMixin
else:
    QuantizationConfig = PretrainedConfig
from enum import Enum


class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    RTN = "rtn"
    AUTOROUND = "autoround"
    TEQ = "teq"


class ITREXQuantizationConfigMixin(QuantizationConfig):
    """
    Mixin class for quantization config
    """

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing atributtes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs

    def post_init_cpu(self):
        r"""
        Safety checker that arguments are correct
        """

        if self.compute_dtype is not None and self.compute_dtype not in [
            "fp32",
            "bf16",
            "int8",
        ]:
            raise ValueError("compute_dtype must be 'fp32', 'bf16', 'int8'.")
        elif self.compute_dtype is None:
            self.compute_dtype = "fp32"

        if self.bits is None:
            self.bits = 4
        elif self.bits is not None and self.bits not in [4, 8]:
            raise ValueError(
                f"Only support quantization to [4, 8] bits but found {self.bits}"
            )

        if self.bits == 4 and self.weight_dtype not in [
            "int4_fullrange",
            "int4_clip",
            "nf4",
            "fp4_e2m1_bnb",
            "fp4_e2m1",
        ]:
            self.weight_dtype = "int4_clip"
            logger.warning(
                "int4_clip weight_type is used due to bits is 4 but weight_dtype is not set."
            )

        if self.bits == 8 and self.weight_dtype not in ["int8", "fp8_e5m2", "fp8_e4m3"]:
            self.weight_dtype = "int8"
            logger.warning(
                "int8 weight_type is used due to bits is 8 but weight_dtype is not set."
            )

        elif self.weight_dtype not in [
            "int8",
            "int4_fullrange",
            "int4_clip",
            "nf4",
            "fp4_e2m1_bnb",
            "fp4_e2m1",
            "fp8_e5m2",
            "fp8_e4m3",
        ]:
            raise ValueError(
                f"weight_dtype must be a string in "
                f"'int8', 'int4_fullrange', 'int4_clip', 'nf4', 'fp4_e2m1_bnb', 'fp4_e2m1', 'fp8_e5m2, fp8_e4m3'"
            )

        if self.scale_dtype is not None and self.scale_dtype not in [
            "fp32",
            "fp8_e8m0",
        ]:
            raise ValueError(
                f"scale_dtype must be a string in 'fp32', 'fp8_e8m0' "
                f"and fp8_e8m0 only used for weight_dtype 'fp8_e5m2', 'fp8_e4m3'"
            )
        elif self.scale_dtype is None:
            self.scale_dtype = "fp32"

        if not isinstance(self.use_double_quant, bool):
            raise ValueError("use_double_quant must be a boolean")

        if self.use_double_quant and not isinstance(self.double_quant_dtype, str):
            raise ValueError("double_quant_dtype must be a string")

        if self.use_double_quant and not isinstance(self.scale_dtype, str):
            raise ValueError("scale_dtype must be a string")

        if not isinstance(self.group_size, int):
            raise ValueError("group_size must be a int")

        if not isinstance(self.scheme, str):
            raise ValueError("scheme must be a string")

        if self.scheme == "asym" and (
            self.compute_dtype == "int8"
            or self.weight_dtype.startswith("fp")
            or self.weight_dtype.startswith("nf")
            or self.scale_dtype != "fp32"
        ):
            raise ValueError(
                "WeightOnlyQuantization doesn't support asym with \
                                compute_dtype int8 or weight_dtype float or scale_dtype non-fp32 now, \
                                please use sym scheme"
            )

        self.use_neural_speed = False

    def post_init_xpu(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

        if self.compute_dtype is not None and self.compute_dtype not in ["fp16"]:
            raise ValueError("compute_dtype must be 'fp16'.")
        elif self.compute_dtype is None:
            self.compute_dtype = "fp16"

        if self.bits is None:
            self.bits = 4
        elif self.bits not in [4]:
            raise ValueError(
                f"Only support quantization to [4] bits but found {self.bits}"
            )

        if self.weight_dtype is None:
            self.weight_dtype = "int4_fullrange"

        elif self.weight_dtype not in [
            "int4_fullrange",
        ]:
            raise ValueError(f"weight_dtype must be a string in 'int4_fullrange', but get {self.weight_dtype}.")

        if self.scale_dtype is not None and self.scale_dtype not in ["fp16"]:
            raise ValueError(f"scale_dtype must be a string in 'fp16'")
        elif self.scale_dtype is None:
            self.scale_dtype = "fp16"

        if not isinstance(self.use_double_quant, bool):
            raise ValueError("use_double_quant must be a boolean")

        if self.use_double_quant and not isinstance(self.double_quant_dtype, str):
            raise ValueError("double_quant_dtype must be a string")

        if self.use_double_quant and not isinstance(self.scale_dtype, str):
            raise ValueError("scale_dtype must be a string")

        if not isinstance(self.group_size, int):
            raise ValueError("group_size must be a int")

        if self.scheme not in ["sym"]:
            raise ValueError(
                "scheme: {} is not support, only support 'sym' now!".format(self.scheme)
            )
        self.use_neural_speed = False

    def post_init_runtime(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

        # MX-compliant format
        # https://arxiv.org/abs/2310.10537
        runtime_supported_compute_dtype = ["fp32", "fp16", "bf16", "int8"]
        runtime_supported_weight_dtype = [
            "int4",
            "int4_clip",  # int4_clip will merge to int4 in next release.
            "int8",
            "fp8",
            "fp8_e5m2",
            "fp8_e4m3",
            "fp4",
            "fp4_e2m1",
            "nf4",
        ]
        runtime_supported_scale_dtype = ["fp32", "bf16", "fp8"]
        runtime_supported_group_size = [-1, 32, 128]
        runtime_supported_scheme = ["sym", "asym"]

        if self.compute_dtype is None:
            self.compute_dtype = "fp32"
        else:
            if self.compute_dtype not in runtime_supported_compute_dtype:
                raise ValueError(
                    "compute_dtype must be in {}.".format(
                        runtime_supported_compute_dtype
                    )
                )

        if self.bits is None:
            self.bits = 4
        elif self.bits not in [4, 8]:
            raise ValueError(
                f"Only support quantization to [4, 8] bits but found {self.bits}"
            )

        if self.weight_dtype is None:
            self.weight_dtype = "int4"
        elif self.weight_dtype == "int4_clip":
            self.weight_dtype == "int4"
        elif self.weight_dtype == "fp8":
            self.weight_dtype == "fp8_e4m3"
        elif self.weight_dtype == "fp8":
            self.weight_dtype == "fp8_e4m3"
        elif self.weight_dtype == "fp4":
            self.weight_dtype = "fp4_e2m1"
        else:
            if self.weight_dtype not in runtime_supported_weight_dtype:
                raise ValueError(
                    "weight_dtype must be in {}.".format(runtime_supported_weight_dtype)
                )

        if self.bits == 4 and self.weight_dtype not in ["int4", "nf4", "fp4_e2m1"]:
            self.weight_dtype = "int4"
            print(
                "int4 weight_type is used due to bits is 4 but weight_dtype is not set."
            )

        if self.bits == 8 and self.weight_dtype not in ["int8", "fp8_e5m2", "fp8_e4m3"]:
            self.weight_dtype = "int8"
            print(
                "int8 weight_type is used due to bits is 8 but weight_dtype is not set."
            )

        if self.scale_dtype is None:
            self.scale_dtype = "fp32"
        else:
            if self.scale_dtype not in runtime_supported_scale_dtype:
                raise ValueError(
                    "scale_dtype must be in {}.".format(runtime_supported_scale_dtype)
                )

        if self.group_size not in runtime_supported_group_size:
            raise ValueError(
                "group_size must be an integer in {}.".format(
                    runtime_supported_group_size
                )
            )

        if self.weight_dtype[:3] in ["fp8", "fp4", "nf4"]:
            if self.compute_dtype in ["int8"]:
                print(
                    "WARNING: int8 compute dtype is not be supported in float quant types! "
                    "Fall back to fp32."
                )
                self.compute_dtype = "fp32"
            if self.scheme in ["asym"]:
                print(
                    "WARNING: asym alg is not be supported in float quant types! "
                    "Fall back to sym."
                )
                self.scheme = "sym"
            if self.scale_dtype in ["fp8"] and self.weight_dtype[:3] not in ["fp8"]:
                print(
                    "WARNING: fp8 scale is only be supported in fp8 weight type. "
                    "Fall back to fp32."
                )
                self.scale_dtype = "fp32"
            if self.weight_dtype[:3] == "fp8" and self.scale_dtype not in [
                "fp8",
                "fp32",
            ]:
                print(
                    "WARNING: fp8 weight type only supports fp8 / fp32 scale now."
                    " Fall back to fp8."
                )
                self.scale_dtype = "fp8"

        self.use_neural_speed = True

    def to_json_file(
        self, json_file_path: Union[str, os.PathLike], use_diff: bool = True
    ):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        # set tokenizer to None due to it doesn't support write to json
        if hasattr(self, "tokenizer"):
            self.tokenizer = None
        if hasattr(self, "calib_dataloader"):
            self.calib_dataloader = None
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def remove_redundant_parameters(self):
        remove_parameters = ["calib_dataloader", "dataset", "calib_func", "calib_iters", "calib_len",
        "double_quant_scale_dtype", "use_double_quant", "mse_range", "scheme", "tokenizer", "use_ggml",
        "use_neural_speed", "use_quant", "layer_wise", "blocksize", "nsamples", "max_input_length", "static_groups",
        "lr", "minmax_lr", "iters", "use_quant_input", "device"]
        for parameter in remove_parameters:
            if hasattr(self, parameter):
                delattr(self, parameter)
        if self.quant_method.value == "awq":
            delattr(self, "sym")

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, QUANT_CONFIG)

        self.to_json_file(output_config_file, use_diff=False)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token", None),
            )

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cf = kwargs.pop("_configuration_file", QUANT_CONFIG)
        return super().get_config_dict(
            pretrained_model_name_or_path, _configuration_file=cf, **kwargs
        )


class RtnConfig(ITREXQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 32,
        compute_dtype: Any = None,
        weight_dtype: Any = None,
        scale_dtype: Any = None,
        mse_range: bool = False,
        use_double_quant=False,
        double_quant_scale_dtype=None,  # reserve for double quant
        sym: bool = True,
        layer_wise: bool = False,
        use_ggml: bool = False,
        use_quant: bool = True,
        use_neural_speed: bool = False,
        llm_int8_skip_modules=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.RTN
        self.bits = bits
        self.mse_range = mse_range
        self.compute_dtype = compute_dtype
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype
        self.group_size = group_size
        self.layer_wise = layer_wise
        self.sym = sym
        self.scheme = "sym" if self.sym else "asym"
        self.use_double_quant = use_double_quant
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.llm_int8_skip_modules = (
            llm_int8_skip_modules if llm_int8_skip_modules else []
        )
        self.use_ggml = use_ggml
        self.use_quant = use_quant
        self.use_neural_speed = use_neural_speed
        self.device = kwargs.get("device", "auto")
        self.calib_dataloader = None
        self.dataset = None
        self.calib_func = None
        self.calib_iters = None

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = RtnConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

class GPTQConfig(ITREXQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 4,
        tokenizer: Any = None,
        dataset: str = "NeelNanda/pile-10k",
        group_size: int = 32,
        compute_dtype: Any = None,
        weight_dtype: Any = None,
        scale_dtype: Any = None,
        use_double_quant=False,
        double_quant_scale_dtype=None,  # reserve for double quant
        sym: bool = True,
        blocksize: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        nsamples: int = 128,
        max_input_length: Optional[int] = None,
        static_groups: bool = False,
        layer_wise: bool = False,
        use_ggml: bool = False,
        use_quant: bool = True,
        use_neural_speed: bool = False,
        llm_int8_skip_modules=None,
        **kwargs,
    ):

        from intel_extension_for_transformers.transformers.llm.quantization.utils import (
            convert_dtype_torch2str,
        )

        self.quant_method = QuantizationMethod.GPTQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.compute_dtype = compute_dtype
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype
        self.sym = sym
        self.use_double_quant = use_double_quant
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.blocksize = blocksize
        self.nsamples = nsamples
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.static_groups = static_groups
        self.layer_wise = layer_wise
        self.max_input_length = max_input_length
        self.llm_int8_skip_modules = (
            llm_int8_skip_modules if llm_int8_skip_modules else []
        )
        self.use_ggml = use_ggml
        self.use_quant = use_quant
        self.use_neural_speed = use_neural_speed
        self.device = kwargs.get("device", "auto")
        self.calib_dataloader = kwargs.get("calib_dataloader", None)
        self.calib_func = kwargs.get("calib_func", None)
        self.calib_iters = kwargs.get("calib_iters", 100)
        self.scheme = "sym" if self.sym else "asym"

        if isinstance(compute_dtype, torch.dtype):
            self.compute_dtype = convert_dtype_torch2str(compute_dtype)
        else:
            self.compute_dtype = compute_dtype

        if isinstance(scale_dtype, torch.dtype):
            self.scale_dtype = convert_dtype_torch2str(scale_dtype)
        else:
            self.scale_dtype = scale_dtype

        if isinstance(double_quant_scale_dtype, torch.dtype):
            self.double_quant_scale_dtype = convert_dtype_torch2str(
                double_quant_scale_dtype
            )
        else:
            self.double_quant_scale_dtype = double_quant_scale_dtype
        self.post_init_gptq()

    def post_init_gptq(self):
        r"""
        Safety checker that arguments are correct
        """

        if self.bits not in [4, 8]:
            raise ValueError(
                f"Only support quantization to [4, 8] bits but found {self.bits}"
            )

        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = GPTQConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

class AwqConfig(ITREXQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 8,
        tokenizer: Any = None,
        dataset: str = "NeelNanda/pile-10k",
        group_size: int = 32,
        compute_dtype: Any = None,
        weight_dtype: Any = None,
        scale_dtype: Any = None,
        use_double_quant=False,
        double_quant_scale_dtype=None,  # reserve for double quant
        zero_point: bool = True,
        mse_range: bool = False,
        use_ggml: bool = False,
        use_quant: bool = True,
        use_neural_speed: bool = False,
        llm_int8_skip_modules=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.AWQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.compute_dtype = compute_dtype
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype
        self.group_size = group_size
        self.zero_point = zero_point
        self.mse_range = mse_range
        self.use_double_quant = use_double_quant
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.llm_int8_skip_modules = (
            llm_int8_skip_modules if llm_int8_skip_modules else []
        )
        self.use_ggml = use_ggml
        self.use_quant = use_quant
        self.use_neural_speed = use_neural_speed
        self.device = kwargs.get("device", "auto")
        self.calib_dataloader = kwargs.get("calib_dataloader", None)
        self.calib_func = kwargs.get("calib_func", None)
        self.calib_iters = kwargs.get("calib_iters", 100)
        self.scheme = "asym" if self.zero_point else "sym"
        self.sym = True if not self.zero_point else False

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = AwqConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

class TeqConfig(ITREXQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 8,
        tokenizer: Any = None,
        dataset: str = "NeelNanda/pile-10k",
        group_size: int = 32,
        compute_dtype: Any = None,
        weight_dtype: Any = None,
        scale_dtype: Any = None,
        use_double_quant=False,
        double_quant_scale_dtype=None,  # reserve for double quant
        sym: bool = True,
        use_ggml: bool = False,
        use_neural_speed: bool = False,
        llm_int8_skip_modules=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.TEQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.compute_dtype = compute_dtype
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype
        self.group_size = group_size
        self.sym = sym
        self.scheme = "sym" if self.sym else "asym"
        self.use_double_quant = use_double_quant
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.llm_int8_skip_modules = (
            llm_int8_skip_modules if llm_int8_skip_modules else []
        )
        self.use_ggml = use_ggml
        self.use_neural_speed = use_neural_speed
        self.device = kwargs.get("device", "auto")
        self.calib_dataloader = kwargs.get("calib_dataloader", None)
        self.calib_func = kwargs.get("calib_func", None)
        self.calib_iters = kwargs.get("calib_iters", 100)

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = TeqConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

class AutoRoundConfig(ITREXQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 8,
        tokenizer: Any = None,
        dataset: str = "NeelNanda/pile-10k",
        group_size: int = 32,
        compute_dtype: Any = None,
        weight_dtype: Any = None,
        scale_dtype: Any = None,
        use_double_quant=False,
        double_quant_scale_dtype=None,  # reserve for double quant
        sym: bool = True,
        lr: float = 0.0025,
        minmax_lr: float = 0.0025,
        use_quant_input: bool = True,
        nsamples: int = 128,
        iters: int = 200,
        use_ggml: bool = False,
        use_neural_speed: bool = False,
        llm_int8_skip_modules=None,
        **kwargs,
    ):

        from intel_extension_for_transformers.transformers.llm.quantization.utils import (
            convert_dtype_torch2str,
        )

        self.quant_method = QuantizationMethod.AUTOROUND
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.compute_dtype = compute_dtype
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype
        self.sym = sym
        self.use_double_quant = use_double_quant
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.nsamples = nsamples
        self.group_size = group_size
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.use_quant_input = use_quant_input
        self.iters = iters
        self.llm_int8_skip_modules = (
            llm_int8_skip_modules if llm_int8_skip_modules else []
        )
        self.use_ggml = use_ggml
        self.use_neural_speed = use_neural_speed
        self.device = kwargs.get("device", "auto")
        self.calib_dataloader = kwargs.get("calib_dataloader", None)
        self.calib_len = kwargs.get("calib_len", None)
        self.calib_func = kwargs.get("calib_func", None)
        self.calib_iters = kwargs.get("calib_iters", 100)
        self.scheme = "sym" if self.sym else "asym"
        if isinstance(compute_dtype, torch.dtype):
            self.compute_dtype = convert_dtype_torch2str(compute_dtype)
        else:
            self.compute_dtype = compute_dtype

        if isinstance(scale_dtype, torch.dtype):
            self.scale_dtype = convert_dtype_torch2str(scale_dtype)
        else:
            self.scale_dtype = scale_dtype

        if isinstance(double_quant_scale_dtype, torch.dtype):
            self.double_quant_scale_dtype = convert_dtype_torch2str(
                double_quant_scale_dtype
            )
        else:
            self.double_quant_scale_dtype = double_quant_scale_dtype

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = AutoRoundConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict
