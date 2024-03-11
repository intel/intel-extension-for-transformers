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

import unittest, os, shutil
from unittest.mock import patch
from intel_extension_for_transformers.neural_chat.tools.evaluation.data_augmentation import retrieval_dataset_construction
from intel_extension_for_transformers.neural_chat.tools.evaluation.retriever import evaluate_retrieval

class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("data.jsonl"):
            os.remove("data.jsonl")
        if os.path.exists("data_minedHN.jsonl"):
            os.remove("data_minedHN.jsonl")
        if os.path.exists("data_minedHN_split.jsonl"):
            os.remove("data_minedHN_split.jsonl")
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("data.jsonl"):
            os.remove("data.jsonl")
        if os.path.exists("data_minedHN.jsonl"):
            os.remove("data_minedHN.jsonl")
        if os.path.exists("data_minedHN_split.jsonl"):
            os.remove("data_minedHN_split.jsonl")
        return super().tearDown()

    def test_retrieval_dataset_construction(self):
        argv = ['--llm_model', '/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3-1', \
                '--embedding_model', '/tf_dataset2/inc-ut/gte-base', \
                '--input', '/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/docs/retrieve_multi_doc/', \
                '--output', 'data', \
                '--range_for_sampling', '2-2', \
                '--negative_number', '1']

        with patch('sys.argv', ['python retrieval_dataset_construction.py'] + argv):
            retrieval_dataset_construction.main()
            self.assertTrue(os.path.exists("data_minedHN_split.jsonl"))
            
    def test_evaluate_retrieval(self):
        argv = ['--index_file_jsonl_path', '/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/candidate_context.jsonl', \
                '--query_file_jsonl_path', '/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/example.jsonl', \
                '--embedding_model', '/tf_dataset2/inc-ut/gte-base']

        with patch('sys.argv', ['python evaluate_retrieval.py'] + argv):
            result = evaluate_retrieval.main()
            self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
