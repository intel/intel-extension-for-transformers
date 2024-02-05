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
    parse_args, warmup, check_args, construct_chatbot
)
from intel_extension_for_transformers.neural_chat.server.restful.openai_protocol import ChatCompletionRequest


class TestMultiCPUServer(unittest.TestCase):

    def test_check_args(self):
        mock_args = MagicMock()
        mock_args.temperature = 0.1
        mock_args.top_p = 0.1
        mock_args.top_k = 1
        mock_args.repetition_penalty = 1.1
        mock_args.num_beams = 1
        mock_args.max_new_tokens = 128
        try:
            check_args(mock_args)
        except ValueError:
            self.fail("Function <check_args()> raised ValueError unexpectedly")

    def test_check_args_value_error(self):
        mock_args = MagicMock()
        mock_args.temperature = 1.5
        mock_args.top_p = 0.1
        mock_args.top_k = 1
        mock_args.repetition_penalty = 1.1
        mock_args.num_beams = 1
        mock_args.max_new_tokens = 128
        with self.assertRaises(ValueError):
            check_args(mock_args)

    @patch("intel_extension_for_transformers.neural_chat.server.multi_cpu_server.build_chatbot")
    def test_construct_chatbot(self, mock_build_chatbot):
        mock_build_chatbot.return_value = MagicMock()
        mock_args = MagicMock()
        mock_args.temperature = 0.1
        mock_args.top_p = 0.1
        mock_args.top_k = 1
        mock_args.repetition_penalty = 1.1
        mock_args.num_beams = 1
        mock_args.max_new_tokens = 128

        chatbot, gen_config = construct_chatbot(mock_args)
        self.assertEqual(gen_config.device, "cpu")
        mock_build_chatbot.assert_called_once()

    def test_warmup(self):
        mock_chatbot = MagicMock()
        mock_chatbot.predict_stream.return_value = [['How ', 'are ', 'you', '?']]

        response = warmup(mock_chatbot, 0, None)
        print(response)
        mock_chatbot.predict_stream.assert_called_once()


if __name__ == "__main__":
    unittest.main()
