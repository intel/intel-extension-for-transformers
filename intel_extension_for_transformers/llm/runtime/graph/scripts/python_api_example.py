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

from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig
model_name = "mosaicml/mpt-7b"
woq_config = WeightOnlyQuantConfig(compute_dtype="int8")

model = AutoModel.from_pretrained(model_name, quantization_config=woq_config)

prompt = "Once upon a time, a little girl"
print(model.generate(prompt, max_new_tokens=30))
