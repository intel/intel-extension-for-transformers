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

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        yaml_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests/ci/server/askdoc.yaml"
        if os.path.exists(yaml_file_path):
            command = f'neuralchat_server start \
                        --config_file {yaml_file_path} \
                        --log_file "./neuralchat.log"'
        else:
            command = 'sed -i "s|askdoc|ci/server/askdoc|g" ./ci/server/askdoc.yaml && neuralchat_server start \
                        --config_file "./ci/server/askdoc.yaml" \
                        --log_file "./neuralchat.log"'
        try:
            self.server_process = subprocess.Popen(command,
                                    universal_newlines=True, shell=True) # nosec
            time.sleep(120)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)

    def tearDown(self) -> None:
        import shutil
        if os.path.exists("./out_persist"):
            shutil.rmtree("./out_persist")

    def test_askdoc_chat(self):
        url = 'http://127.0.0.1:6000/v1/aiphotos/askdoc/chat'
        request = {
            "query": "What is Intel oneAPI Compiler?",
            "knowledge_base_id": "default",
            "stream": False,
            "max_new_tokens": 256
        }
        res = requests.post(url, json.dumps(request))
        self.assertEqual(res.status_code, 200)

    def test_askdoc_upload_link(self):
        url = 'http://127.0.0.1:6000/v1/aiphotos/askdoc/upload_link'
        request = {
            "link_list": ["https://www.ces.tech/"],
            "knowledge_base_id": "default"
        }
        res = requests.post(url, json.dumps(request))
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
