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
import json
import requests
from pathlib import Path

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        yaml_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests/ci/server/plugin_as_service.yaml"
        if os.path.exists(yaml_file_path):
            command = f'neuralchat_server start \
                        --config_file {yaml_file_path} \
                        --log_file "./neuralchat.log"'
        elif os.path.exists("./plugin_as_service.yaml"):
            command = f'neuralchat_server start \
                                    --config_file ./plugin_as_service.yaml \
                                    --log_file "./neuralchat.log"'
        else:
            command = 'sed -i "s|plugin_as_service|ci/server/plugin_as_service|g" \
                        ./ci/server/plugin_as_service.yaml && neuralchat_server start \
                        --config_file "./ci/server/plugin_as_service.yaml" \
                        --log_file "./neuralchat.log"'
        try:
            self.server_process = subprocess.Popen(command,
                                    universal_newlines=True, shell=True) # nosec
            time.sleep(30)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)


    def test_plugin_as_service(self):
        url = 'http://127.0.0.1:7777/plugin/audio/asr'
        file_path = Path("/intel-extension-for-transformers/intel_extension_for_transformers \
                         /neural_chat/assets/audio/welcome.wav")
        
        if file_path.is_file():
            with file_path.open("rb") as file:
                response = requests.post(url, files={"file": file})
            print(response.text)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text.lower(), "welcome to neuralchat")
        else:
            print(f"No such file: {file_path}")

        url = 'http://127.0.0.1:7777/plugin/audio/tts'
        request = {
            "text": "Hello",
            "voice": "default",
            "knowledge_id": "default"
        }
        res = requests.post(url, json.dumps(request))
        self.assertEqual(res.status_code, 200)

if __name__ == "__main__":
    unittest.main()
