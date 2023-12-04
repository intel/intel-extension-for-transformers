#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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


from .config import (
    WEIGHTS_NAME,
    AutoDistillationConfig,
    BenchmarkConfig,
    DistillationConfig,
    DynamicLengthConfig,
    FlashDistillationConfig,
    NASConfig,
    Provider,
    PrunerV2,
    PruningConfig,
    QuantizationConfig,
    TFDistillationConfig,
)
from .distillation import (
    SUPPORTED_DISTILLATION_CRITERION_MODE,
    DistillationCriterionMode,
)
from .mixture.auto_distillation import AutoDistillation
from .nas import NAS
from .optimizer import NoTrainerOptimizer, Orchestrate_optimizer
from .optimizer_tf import TFOptimization
from .pruning import SUPPORTED_PRUNING_MODE, PrunerConfig, PruningMode
from .quantization import SUPPORTED_QUANT_MODE, QuantizationMode
from .utils import (
    MixedPrecisionConfig,
    BitsAndBytesConfig,
    SmoothQuantConfig,
    WeightOnlyQuantConfig,
    metrics,
    objectives,
)
from .utils.utility import LazyImport
from .modeling import (
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    OptimizedModel
)
