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

"""Quantization: specify the supported quantization mode."""

from enum import Enum
from transformers.utils.versions import require_version

try:
    require_version("neural_compressor>=1.9.0")
except:
    require_version("neural_compressor_full>=1.9.0", "To fix: pip install neural_compressor")


class QuantizationMode(Enum):
    """Currently support three quantization modes."""
    POSTTRAININGSTATIC = "post_training_static_quant"
    POSTTRAININGDYNAMIC = "post_training_dynamic_quant"
    QUANTIZATIONAWARETRAINING = "quant_aware_training"


SUPPORTED_QUANT_MODE = set([approach.name for approach in QuantizationMode])
