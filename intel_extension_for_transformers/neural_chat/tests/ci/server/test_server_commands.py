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
from intel_extension_for_transformers.neural_chat.server.server_commands import (
    get_server_command, NeuralChatServerHelpCommand,
    get_client_command, NeuralChatClientHelpCommand,
    neuralchat_server_execute, neuralchat_client_execute,
    NeuralChatClientBaseCommand
)


class TestServerCommand(unittest.TestCase):

    def test_get_server_command(self):
        res = get_server_command('neuralchat_server.help')
        self.assertIs(res, NeuralChatServerHelpCommand)

    def test_get_client_command(self):
        res = get_client_command('neuralchat_client.help')
        self.assertIs(res, NeuralChatClientHelpCommand)

    def test_neuralchat_server_execute(self):
        res = neuralchat_server_execute()
        self.assertEqual(res, 0)

    def test_neuralchat_client_execute(self):
        res = neuralchat_client_execute()
        self.assertEqual(res, 0)


if __name__ == "__main__":
    unittest.main()
