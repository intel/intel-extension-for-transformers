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

import shutil
import torch
import unittest
import neural_compressor.adaptor.pytorch as nc_torch
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
PT_VERSION = nc_torch.get_torch_version()

class TestVLLM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("./runtime_outs", ignore_errors=True)

    @unittest.skipIf(PT_VERSION.release < Version("2.3.0").release,
            "Please use PyTroch 2.3.0 or higher version for vllm")
    def test_use_vllm_api(self):
        model_name = "/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
        prompt = "Once upon a time"
        model = AutoModelForCausalLM.from_pretrained(model_name, use_vllm=True)
        output = model.generate(prompt)
        print("output = ", output)


if __name__ == "__main__":
    unittest.main()
