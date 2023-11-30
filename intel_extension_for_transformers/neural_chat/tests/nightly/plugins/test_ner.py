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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.ner.ner import NamedEntityRecognition
from intel_extension_for_transformers.neural_chat.pipeline.plugins.ner.ner_int import NamedEntityRecognitionINT
import unittest
import os


class TestNER(unittest.TestCase):
    def setUp(self):
        self.skipTest("skip before debug finish")
        return super().setUp()

    def tearDown(self) -> None:
        for filename in os.getcwd():
            import re
            if re.match(r'ne_.*_fp32.bin', filename) or re.match(r'ne_.*_q.bin', filename):
                file_path = os.path.join(os.getcwd(), filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {filename}")
                except OSError as e:
                    print(f"Error deleting file {filename}: {str(e)}")
        return super().tearDown()

    def test_ner(self):
        os.system('python -m spacy download en_core_web_lg')
        ner_obj = NamedEntityRecognition()
        query = "Show me photos taken in Shanghai."
        result = ner_obj.ner_inference(query)
        _result = {
            'period': [], 
            'time': [], 
            'location': ['Shanghai'], 
            'name': [], 
            'organization': []
        }
        self.assertEqual(result, _result)


if __name__ == "__main__":
    unittest.main()