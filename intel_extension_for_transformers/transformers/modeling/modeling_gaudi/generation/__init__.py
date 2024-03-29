# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .configuration_utils import GaudiGenerationConfig
from .stopping_criteria import (
    gaudi_MaxLengthCriteria_call,
    gaudi_MaxNewTokensCriteria_call,
)
from .utils import MODELS_OPTIMIZED_WITH_STATIC_SHAPES, GaudiGenerationMixin
