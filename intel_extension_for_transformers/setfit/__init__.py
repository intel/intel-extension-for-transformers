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

from .modeling import SetFitHead, SetFitModel
from .trainer import SetFitTrainer
from .trainer_distillation import DistillationSetFitTrainer
from intel_extension_for_transformers.transformers.utils.utility import LazyImport

setfit = LazyImport("setfit")
add_templated_examples = setfit.add_templated_examples
get_templated_dataset = setfit.get_templated_dataset
sample_dataset = setfit.sample_dataset
