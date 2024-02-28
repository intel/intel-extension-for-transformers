#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
import unittest

class TestBioGPTModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_run_inference(self):
        config = PipelineConfig(
            model_name_or_path="/tf_dataset2/models/nlp_toolkit/biogpt")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("COVID-19 is ")
        print(result)
        self.assertIn('COVID-19 is', str(result)) and self.assertIn('pandemic', str(result))

if __name__ == "__main__":
    unittest.main()
