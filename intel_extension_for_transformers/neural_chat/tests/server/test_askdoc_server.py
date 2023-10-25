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
            "intel_extension_for_transformers/neural_chat/tests/server/askdoc.yaml"
        if os.path.exists(yaml_file_path):
            command = f'neuralchat_server start \
                        --config_file {yaml_file_path} \
                        --log_file "./neuralchat.log"'
        else:
            command = 'neuralchat_server start \
                        --config_file "./askdoc.yaml" \
                        --log_file "./neuralchat.log"'
        try:
            self.server_process = subprocess.Popen(command,
                                    universal_newlines=True, shell=True) # nosec
            time.sleep(30)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)

    def tearDown(self) -> None:
        import shutil
        if os.path.exists("./out_persist"):
            shutil.rmtree("./out_persist")
        os.system("ps -ef |grep 'askdoc.yaml' |awk '{print $2}' |xargs kill -9")

    def test_askdoc_chat(self):
        url = 'http://127.0.0.1:9000/v1/askdoc/chat'
        request = {
            "query": "What is Intel oneAPI Compiler?",
            "domain": "test",
            "blob": "",
            "filename": ""
        }
        res = requests.post(url, json.dumps(request))
        self.assertEqual(res.status_code, 200)

    def test_askdoc_upload(self):
        file_name = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests/server/askdoc/test_doc.txt"
        if not os.path.exists(file_name):
            file_name = "./askdoc/test_doc.txt"
        if not os.path.exists("/home/sdp/askdoc_upload/enterprise_docs"):
            os.mkdir("/home/sdp/askdoc_upload/enterprise_docs")
        url = 'http://127.0.0.1:9000/v1/askdoc/upload'
        with open(file_name, "r") as upload_file:
            res = requests.post(url, files={'file': upload_file})
            print(res.text)
            self.assertEqual(res.text, '{"knowledge_base_id":"fake_knowledge_base_id"}')

if __name__ == "__main__":
    unittest.main()
