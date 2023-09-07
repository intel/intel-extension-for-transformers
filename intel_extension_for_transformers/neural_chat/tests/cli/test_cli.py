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

import os
import unittest
from intel_extension_for_transformers.neural_chat.cli.log import logger
import subprocess

class UnitTest(unittest.TestCase):

    def test_text_chat(self):
        logger.info(f'Testing CLI request === Text Chat ===')
        command = 'neuralchat textchat \
                    --query "Tell me about Intel." \
                    --model_name_or_path "facebook/opt-125m"'
        result = None
        try:
            result = subprocess.run(command, capture_output = True, check=True,
                                    universal_newlines=True, shell=True) # nosec
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.assertIn("model loaded", result.stdout)

    def test_help(self):
        logger.info(f'Testing CLI request === Help ===')
        command = 'neuralchat help'
        result = None
        try:
            result = subprocess.run(command, capture_output = True, check=True,
                                    universal_newlines=True, shell=True) # nosec
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.assertIn("Show help for neuralchat commands.", result.stdout)

    def test_voice_chat(self):
        logger.info(f'Testing CLI request === Voice Chat ===')
        command = 'neuralchat voicechat \
                    --query "Tell me about Intel Xeon Scalable Processors." \
                    --audio_output_path "./response.wav" \
                    --model_name_or_path "facebook/opt-125m"'
        result = None
        try:
            result = subprocess.run(command, capture_output = True, check=True,
                                    universal_newlines=True, shell=True) # nosec
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.assertIn("model loaded", result.stdout)


if __name__ == "__main__":
    unittest.main()