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
            if re.match(r'ne_.*_fp32.bin', filename) or re.match(r'ne_.*_q.bin', filename):
                file_path = os.path.join(os.getcwd(), filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {filename}")
                except OSError as e:
                    print(f"Error deleting file {filename}: {str(e)}")
        return super().tearDown()

    def test_fp32(self):
        os.system('python -m spacy download en_core_web_lg')
        ner_obj = NamedEntityRecognition(model_path="/tf_dataset2/models/nlp_toolkit/mpt-7b")
        query = "Show me photos taken in Shanghai."
        result = ner_obj.inference(query=query)
        _result = {
            'period': [], 
            'time': [], 
            'location': ['Shanghai'], 
            'name': [], 
            'organization': []
        }
        self.assertEqual(result, _result)

    def test_bf16(self):
        os.system('python -m spacy download en_core_web_lg')
        ner_obj = NamedEntityRecognition(model_path="/tf_dataset2/models/nlp_toolkit/mpt-7b", bf16=True)
        query = "Show me photos taken in Shanghai."
        result = ner_obj.inference(query=query)
        _result = {
            'period': [], 
            'time': [], 
            'location': ['Shanghai'], 
            'name': [], 
            'organization': []
        }
        self.assertEqual(result, _result)

    def test_int8(self):
        os.system('python -m spacy download en_core_web_lg')
        ner_obj = NamedEntityRecognitionINT(model_path="/tf_dataset2/models/nlp_toolkit/mpt-7b")
        query = "Show me photos taken in Shanghai."
        result = ner_obj.inference(query=query, threads=8)
        _result = {
            'period': [], 
            'time': [], 
            'location': ['Shanghai'], 
            'name': [], 
            'organization': []
        }
        self.assertEqual(result, _result)

    def test_int4(self):
        os.system('python -m spacy download en_core_web_lg')
        ner_obj = NamedEntityRecognitionINT(model_path="/tf_dataset2/models/nlp_toolkit/mpt-7b", 
                                            compute_dtype="int8", 
                                            weight_dtype="int4")
        query = "Show me photos taken in Shanghai."
        result = ner_obj.inference(query=query, threads=8)
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