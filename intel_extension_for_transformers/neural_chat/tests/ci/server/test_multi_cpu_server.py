# !/usr/bin/env python
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
from unittest.mock import patch, MagicMock
from intel_extension_for_transformers.neural_chat.server.multi_cpu_server import (
    parse_args, prepare_params, warmup
)


@patch('intel_extension_for_transformers.neural_chat.server.multi_cpu_server.build_chatbot')
class TestMultiCPUServer(unittest.TestCase):
    
    def test_parse_args(self, mock_build_chatbot):
        args = parse_args()
        print(args)
        self.assertEqual(args.temperature, 0.1)

    def test_prepare_params_warmup(self, mock_build_chatbot):
        mock_chatbot = MagicMock()
        mock_chatbot.predict_stream.return_value = [['How ', 'are ', 'you', '?']]
        mock_build_chatbot.return_value = mock_chatbot
        args, config, chatbot, gen_config = prepare_params()
        self.assertEqual(config.device, 'cpu')

        try:
            warmup(args, chatbot, gen_config)
        except Exception as e:
            raise Exception(e)


if __name__ == "__main__":
    unittest.main()
