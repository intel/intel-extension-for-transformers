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

from intel_extension_for_transformers.transformers import AutoModelForCausalLM

prompt = "Once upon a time, a little girl"
model_name = "/mnt/disk1/data2/zhenweil/models/mpt/mpt-7b/"

model = AutoModelForCausalLM.from_pretrained(model_name, use_llm_runtime=True) # weightonlyconfig
print(model.generate("Hello, my dog is cute")) #max_length=100 , sentence_mode=True
