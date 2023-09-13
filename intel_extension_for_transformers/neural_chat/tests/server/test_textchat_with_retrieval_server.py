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
import time
import os
from intel_extension_for_transformers.neural_chat.server import TextChatClientExecutor

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        yaml_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests/server/textchat_with_retrieval.yaml"
        if os.path.exists(yaml_file_path):
            command = f'neuralchat_server start \
                        --config_file {yaml_file_path} \
                        --log_file "./neuralchat.log"'
        else:
            command = 'neuralchat_server start \
                        --config_file "./textchat_with_retrieval.yaml" \
                        --log_file "./neuralchat.log"'
        try:
            self.server_process = subprocess.Popen(command,
                                    universal_newlines=True, shell=True) # nosec
            time.sleep(80)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.client_executor = TextChatClientExecutor()

    def tearDown(self) -> None:
        try:
            # Send SIGTERM (signal 15) to the process group
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            # Wait for a reasonable amount of time for the process to terminate
            self.server_process.wait(timeout=10)

            # If it didn't terminate within the timeout, send SIGKILL (signal 9)
            if self.server_process.poll() is None:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                self.server_process.wait()
        except subprocess.TimeoutExpired:
            # Handle the case where the process did not terminate within the timeout
            print("Process did not terminate within the timeout.")

    def test_text_chat(self):
        result = self.client_executor(
            prompt="Tell me effective Post-Training Quantization for Large Language Models.",
            server_ip="127.0.0.1",
            port=8000)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(os.path.exists("output"))

if __name__ == "__main__":
    unittest.main()
