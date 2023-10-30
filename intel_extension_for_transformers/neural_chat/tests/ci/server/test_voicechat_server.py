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

import subprocess
import unittest
import os
import time
from intel_extension_for_transformers.neural_chat.server import VoiceChatClientExecutor

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        yaml_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests/ci/server/voicechat.yaml"
        if os.path.exists(yaml_file_path):
            command = f'neuralchat_server start \
                        --config_file {yaml_file_path} \
                        --log_file "./neuralchat.log"'
        else:
            command = 'neuralchat_server start \
                        --config_file "./voicechat.yaml" \
                        --log_file "./neuralchat.log"'
        try:
            self.server_process = subprocess.Popen(command,
                                    universal_newlines=True, shell=True) # nosec
            time.sleep(30)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.client_executor = VoiceChatClientExecutor()
    
    def tearDown(self) -> None:
        for filename in os.listdir("."):
            if filename.endswith(".wav"):
                os.remove(filename)

    def test_voice_chat(self):
        audio_path = \
           "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
        if os.path.exists(audio_path):
            self.client_executor(
                audio_input_path=audio_path,
                server_ip="127.0.0.1",
                port=9000)
        else:
            self.client_executor(
                audio_input_path="../../assets/audio/sample.wav",
                server_ip="127.0.0.1",
                port=9000)
        self.assertEqual(os.path.exists("audio_0.wav"), True)

if __name__ == "__main__":
    unittest.main()
