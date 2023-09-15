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

from dataclasses import dataclass
from typing import Optional, Any
from transformers import BitsAndBytesConfig


@dataclass
class WeightOnlyQuantizationConfig:
    algorithm: str = 'RTN'
    bits: int = 8
    group_size: int = -1
    scheme: str = 'sym'
    enable_full_range: bool = True

@dataclass
class AMPConfig:
    dtype: str = 'bfloat16'    

@dataclass
class SmoothQuantConfig:
    tokenizer: Any = None
    calib_func: Any = None
    alpha: float = 0.5
    op_type_dict: dict = None  
    excluded_precisions: dict = None

