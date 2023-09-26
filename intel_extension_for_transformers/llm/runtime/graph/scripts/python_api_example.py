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

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig

model_name = "THUDM/chatglm2-6b"  # or local path to model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
prompt = "She opened the door and see"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

model = AutoModel.from_pretrained(model_name, quantization_config=woq_config)
gen_tokens = model.generate(input_ids.tolist()[0], max_new_tokens=30)
=======
=======
print(input_ids)
>>>>>>> update
=======
>>>>>>> update
streamer = TextStreamer(tokenizer)

<<<<<<< HEAD
model = AutoModel.from_pretrained(model_name, quantization_config=woq_config, use_llm_runtime=True)
<<<<<<< HEAD
gen_tokens = model.generate(input_ids, streamer=streamer, max_new_tokens=30)
>>>>>>> add streamer
=======
=======
model = AutoModel.from_pretrained(model_name, quantization_config=woq_config, use_llm_runtime=True, trust_remote_code=True)
>>>>>>> use chatglm2
gen_tokens = model.generate(input_ids, streamer=streamer, max_new_tokens=300)
>>>>>>> update streamer mode

