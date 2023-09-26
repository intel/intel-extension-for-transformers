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
from typing import Any, Optional, Dict, Union
from .utility import LazyImport
from transformers import BitsAndBytesConfig
torch = LazyImport("torch")

class WeightOnlyQuantConfig:
    def __init__(
        self,
        llm_int8_skip_modules=None,
        compute_dtype=None,
        weight_dtype=None,
        scale_dtype="fp32", # Now only fp32
        mse_range=False,
        use_double_quant=False,
        double_quant_dtype="int8", # reserve for double quant
        double_quant_scale_dtype="fp32", # reserve for double quant
        group_size=32,
        scheme="sym",
        algorithm="RTN",
        use_ggml=False,
        **kwargs,
    ):
        from intel_extension_for_transformers.llm.quantization.utils import convert_dtype_2_str
        self.llm_int8_skip_modules = llm_int8_skip_modules if llm_int8_skip_modules else []
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype
        self.mse_range = mse_range
        self.use_double_quant = use_double_quant
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.scheme = scheme
        self.algorithm = algorithm
        self.group_size = group_size
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.calib_func = kwargs.pop("calib_func", None)
        self.calib_dataset = kwargs.pop("calib_dataset", "NeelNanda/pile-10k")
        self.calib_dataloader = kwargs.pop("calib_dataloader", None)
        self.calib_iters = kwargs.pop("calib_iters", 100)
        self.use_ggml = use_ggml

        if compute_dtype is None:
            self.compute_dtype = "fp32"
        elif isinstance(compute_dtype, str):
            self.compute_dtype = compute_dtype
        elif isinstance(compute_dtype, torch.dtype):
            self.compute_dtype = convert_dtype_2_str(compute_dtype)
        else:
            raise ValueError("bit4_compute_dtype must be a string or a torch.dtype")


    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise ValueError("llm_int8_skip_modules must be a list of strings")

        if self.compute_dtype is not None and self.compute_dtype not in ['fp32', 'bf16', 'int8']:
            raise ValueError("compute_dtype must be 'fp32', 'bf16', 'int8'.")

        if self.weight_dtype is None:
            self.weight_dtype = 'int4_fullrange'
        elif self.weight_dtype not in ['int8', 'int4_fullrange', 'int4_clip', 'nf4', 'fp4_e2m1_bnb', 'fp4_e2m1']:
            raise ValueError(f"weight_dtype must be a string in "
                             f"'int8', 'int4_fullrange', 'int4_clip', 'nf4', 'fp4_e2m1_bnb', 'fp4_e2m1'")

        if self.scale_dtype not in ["fp32"]:
            raise ValueError("scale_dtype must be a string in 'fp32'")

        if not isinstance(self.mse_range, bool):
            raise ValueError("mse_range must be a boolean")

        if not isinstance(self.use_double_quant, bool):
            raise ValueError("use_double_quant must be a boolean")

        if self.use_double_quant and not isinstance(self.double_quant_dtype, str):
            raise ValueError("double_quant_dtype must be a string")

        if self.use_double_quant and not isinstance(self.double_quant_scale_dtype, str):
            raise ValueError("double_quant_scale_dtype must be a string")

        if not isinstance(self.group_size, int):
            raise ValueError("group_size must be a int")

        if not isinstance(self.scheme, str):
            raise ValueError("scheme must be a string")

    def post_init_runtime(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise ValueError("llm_int8_skip_modules must be a list of strings")

        if self.compute_dtype is None:
            self.compute_dtype = "int8"
        elif self.compute_dtype not in ['int8', 'bf16', 'fp32']:
            raise ValueError("compute_dtype must be 'int8', 'bf16', 'fp32'.")

        if self.weight_dtype is None:
            self.weight_dtype = "int4"
        elif self.weight_dtype not in ['int4', 'int8']:
            raise ValueError(f"weight_dtype must be 'int4', 'int8'.")

        if self.scale_dtype not in ["fp32", "fp16"]:
            raise ValueError("scale_dtype must be 'fp32', 'fp16'.")

        if self.group_size not in [32, 128]:
            raise ValueError("group_size must be an integer in [32, 128]")

        if self.scheme not in ["sym", "asym"]:
            raise ValueError("scheme must be 'sym', 'asym'.")

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model.
        """
        # TODO: For training only
        pass

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs, **kwargs):
        """
        Instantiates a [`WeightOnlyQuantConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`WeightOnlyQuantConfig`]: The configuration object instantiated from those parameters.
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

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

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
                `WeightOnlyQuantConfig()`
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
        default_config_dict = WeightOnlyQuantConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@dataclass
class MixedPrecisionConfig:
    dtype: str = "bfloat16"

@dataclass
class SmoothQuantConfig:
    tokenizer: Any = None
    calib_func: Any = None
    calib_dataset: str = "NeelNanda/pile-10k"
    calib_iters: int = 100
    alpha: float = 0.5
    op_type_dict: dict = None
    excluded_precisions: list = field(default_factory=list)
