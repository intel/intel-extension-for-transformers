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



import unittest
import os
import shutil
from unittest import mock
from intel_extension_for_transformers.neural_chat.models.model_utils import load_model, MODELS
from intel_extension_for_transformers.transformers import MixedPrecisionConfig, BitsAndBytesConfig, WeightOnlyQuantConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TestModelUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path = "/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3-1"
        return super().setUpClass()

    def tearDown(self) -> None:
        if os.path.exists("nc_workspace"):
            shutil.rmtree("nc_workspace")
        if os.path.exists("runtime_outs"):
            shutil.rmtree("runtime_outs")
        return super().tearDown()

    @unittest.skipIf(get_device_type() != 'hpu', "Only run this test on HPU")
    def test_load_model_on_hpu(self):
        load_model(model_name=self.model_path, tokenizer_name=self.model_path, device="hpu", use_hpu_graphs=True)
        self.assertTrue(self.model_path in MODELS)
        self.assertTrue(MODELS[self.model_path]["model"] is not None)

    @unittest.skipIf(get_device_type() != 'hpu', "Only run this test on HPU")
    def test_load_model_on_hpu_with_deepspeed(self):
        load_model(model_name=self.model_path, tokenizer_name=self.model_path, device="hpu", use_hpu_graphs=True, use_deepspeed=True)
        self.assertTrue(self.model_path in MODELS)
        self.assertTrue(MODELS[self.model_path]["model"] is not None)

if __name__ == '__main__':
    unittest.main()
