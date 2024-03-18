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

from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat.config import LoadingModelConfig
from intel_extension_for_transformers.transformers import GPTQConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
import unittest

class TestLlama2GPTQModel(unittest.TestCase):
    def setUp(self):
        self.device = get_device_type()
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_code_gen_with_gguf(self):
        if self.device == "hpu":
            self.skipTest("GTPQ is not supported on HPU.")
        loading_config = LoadingModelConfig(use_neural_speed=True)
        optimization_config = GPTQConfig(bits=4)
        config = PipelineConfig(model_name_or_path="/tf_dataset2/models/nlp_toolkit/Llama-2-7B-Chat-GPTQ",
                                optimization_config=optimization_config,
                                loading_config=loading_config)
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('Intel Xeon Scalable Processors', str(result))

if __name__ == "__main__":
    unittest.main()
