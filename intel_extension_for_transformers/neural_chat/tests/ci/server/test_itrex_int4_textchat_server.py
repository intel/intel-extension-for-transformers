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
import json
from intel_extension_for_transformers.neural_chat.server import TextChatClientExecutor

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        yaml_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests/ci/server/textchat_itrex_int4.yaml"
        if os.path.exists(yaml_file_path):
            command = f'neuralchat_server start \
                        --config_file {yaml_file_path} \
                        --log_file "./neuralchat.log"'
        else:
            command = 'neuralchat_server start \
                        --config_file "./ci/server/textchat_itrex_int4.yaml" \
                        --log_file "./neuralchat.log"'
        try:
            self.server_process = subprocess.Popen(command,
                                    universal_newlines=True, shell=True) # nosec
            time.sleep(30)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.client_executor = TextChatClientExecutor()

    def test_text_chat(self):
        result = self.client_executor(
            prompt="Tell me about Intel Xeon processors.",
            server_ip="127.0.0.1",
            port=8080)
        self.assertEqual(result.status_code, 200)
        print(json.loads(result.text))

        result = self.client_executor(
            prompt="Tell me about Intel Xeon processors.",
            server_ip="127.0.0.1",
            port=8080,
            stream=True)
        self.assertEqual(result.status_code, 200)
        for chunk in result.iter_lines(decode_unicode=False, delimiter=b"\0"):
            print(chunk)


if __name__ == "__main__":
    unittest.main()
