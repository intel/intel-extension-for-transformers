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
import os
from unittest.mock import patch, MagicMock
from yacs.config import CfgNode
from intel_extension_for_transformers.neural_chat.server.neuralchat_server import NeuralChatServerExecutor, app, get_config
from intel_extension_for_transformers.neural_chat import plugins

def build_fake_yaml_basic():
    fake_yaml = """
host: 0.0.0.0
port: 8000

model_name_or_path: "facebook/opt-125m"
device: "auto"

# task choices = ['textchat', 'voicechat', 'retrieval', 'text2image', 'finetune'， 'photoai']
tasks_list: ['textchat']
    """
    with open('neuralchat.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

class TestNeuralChatServerExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = NeuralChatServerExecutor()
        build_fake_yaml_basic()

    def tearDown(self):
        os.remove('neuralchat.yaml')

    def test_init_plugin_as_service(self):
        config = CfgNode()
        config.host = '127.0.0.1'
        config.port = 8000
        config.use_deepspeed = True
        config.plugin_as_service = True
        config.device = 'cpu'
        config.model_name_or_path = 'facebook/opt-125m'
        config.tasks_list = ['textchat']
        with patch.dict(plugins, {}):
            self.assertTrue(self.executor.init(config))

    def test_init_chatbot_as_service(self):
        config = CfgNode()
        config.host = '127.0.0.1'
        config.port = 8000
        config.device = 'cpu'
        config.model_name_or_path = 'facebook/opt-125m'
        config.tasks_list = ['textchat']
        with patch.dict(plugins, {}):
            self.assertTrue(self.executor.init(config))

    def test_init_chatbot_as_service_with_deepspeed(self):
        config = CfgNode()
        config.host = '127.0.0.1'
        config.port = 8000
        config.use_deepspeed = True
        config.device = 'cpu'
        config.model_name_or_path = 'facebook/opt-125m'
        config.tasks_list = ['textchat']
        with patch.dict(plugins, {}):
            self.assertTrue(self.executor.init(config))

    def test_init_chatbot_as_service_with_deepspeed_habana(self):
        config = CfgNode()
        config.host = '127.0.0.1'
        config.port = 8000
        config.use_deepspeed = True
        config.device = 'hpu'
        config.model_name_or_path = 'facebook/opt-125m'
        config.tasks_list = ['textchat']
        with patch.dict(plugins, {}):
            self.assertTrue(self.executor.init(config))

    def test_execute_success(self):
        argv = ['--config_file', 'neuralchat.yaml', '--log_file', 'app.log']

        with patch('intel_extension_for_transformers.neural_chat.server.neuralchat_server.get_config') as mock_get_config, \
             patch('intel_extension_for_transformers.neural_chat.server.neuralchat_server.NeuralChatServerExecutor.init') as mock_init, \
             patch('uvicorn.run') as mock_run:

            mock_get_config.return_value = MagicMock()
            mock_init.return_value = True
            self.executor.execute(argv)
            mock_init.assert_called_once_with(mock_get_config.return_value)
            mock_run.assert_called_once_with(app, host=mock_get_config.return_value.host,
                                             port=mock_get_config.return_value.port)

    def test_execute_failure(self):
        argv = ['--config_file', 'neuralchat.yaml', '--log_file', 'app.log']

        with patch('intel_extension_for_transformers.neural_chat.server.neuralchat_server.get_config') as mock_get_config, \
             patch('intel_extension_for_transformers.neural_chat.server.neuralchat_server.NeuralChatServerExecutor.init') as mock_init, \
             patch('uvicorn.run') as mock_run:

            mock_get_config.return_value = MagicMock()
            mock_init.return_value = False
            self.executor.execute(argv)
            mock_init.assert_called_once_with(mock_get_config.return_value)
            mock_run.assert_not_called()

    @patch('uvicorn.run')
    def test_execute_exception(self, mock_run):
        mock_run.side_effect = Exception('Uvicorn run failed simulation for unit test')
        self.executor.execute(['--config_file', 'neuralchat.yaml', '--log_file', 'app.log'])
        mock_run.assert_called_once()

    @patch.object(NeuralChatServerExecutor, '__call__')
    def test_execute_exception_call(self, mock_call):
        mock_call.side_effect = Exception('Failed to start server simulation for unit test')
        with self.assertRaises(SystemExit) as cm:
            self.executor.execute(['--config_file', 'neuralchat.yaml', '--log_file', 'app.log'])

        self.assertEqual(cm.exception.code, -1)
        mock_call.assert_called_once_with('neuralchat.yaml', 'app.log')

if __name__ == '__main__':
    unittest.main()
