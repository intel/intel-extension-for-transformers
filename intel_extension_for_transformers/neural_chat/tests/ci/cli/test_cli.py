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
        command = 'neuralchat predict \
                    --query "Tell me about Intel." \
                    --model_name_or_path "facebook/opt-125m"'
        result = None
        try:
            result = subprocess.run(command, capture_output = True, check=True,
                                    universal_newlines=True, shell=True) # nosec
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.assertIn("Loading model", result.stdout)

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
        audio_path = \
           "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
        if os.path.exists(audio_path):
            command = f'neuralchat predict \
                        --query {audio_path} \
                        --model_name_or_path "facebook/opt-125m"'
        else:
            command = f'neuralchat predict \
                        --query "../assets/audio/sample.wav" \
                        --model_name_or_path "facebook/opt-125m"'
        result = None
        try:
            result = subprocess.run(command, capture_output = True, check=True,
                                    universal_newlines=True, shell=True) # nosec
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.assertIn("Loading model", result.stdout)


if __name__ == "__main__":
    unittest.main()
