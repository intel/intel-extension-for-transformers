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
from intel_extension_for_transformers.neural_chat.tools.evaluation.data_augmentation import retrieval_dataset_construction, llm_generate_truth
from intel_extension_for_transformers.neural_chat.tools.evaluation.retriever import evaluate_retrieval
from intel_extension_for_transformers.neural_chat.tools.evaluation.framework import ragas_evaluation

class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("data"):
            shutil.rmtree("data", ignore_errors=True)
        if os.path.exists("ground_truth.jsonl"):
            os.remove("ground_truth.jsonl")
        if os.path.exists("output"):
            shutil.rmtree("output", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("data"):
            shutil.rmtree("data", ignore_errors=True)
        if os.path.exists("ground_truth.jsonl"):
            os.remove("ground_truth.jsonl")
        if os.path.exists("output"):
            shutil.rmtree("output", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_dataset_construction(self):
        path = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/docs/retrieve_multi_doc/"
        if os.path.exists(path):
            input_path=path
        else:
            input_path='../assets/docs/retrieve_multi_doc/'
        argv = ['--llm_model', '/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3-1', \
                '--embedding_model', '/tf_dataset2/inc-ut/gte-base', \
                '--input', input_path, \
                '--output', './data', \
                '--range_for_sampling', '2-2', \
                '--negative_number', '1']
        with patch('sys.argv', ['python retrieval_dataset_construction.py'] + argv):
            retrieval_dataset_construction.main()
            self.assertTrue(os.path.exists("./data/minedHN_split.jsonl"))

    def test_llm_generate_truth(self):
        path = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/example.jsonl"
        if os.path.exists(path):
            input_path=path
        else:
            input_path='../tools/evaluation/data_augmentation/example.jsonl'
        argv = ['--llm_model', '/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3-1', \
                '--input', input_path, \
                '--output', 'ground_truth.jsonl']
        with patch('sys.argv', ['python llm_generate_truth.py'] + argv):
            llm_generate_truth.main()
            self.assertTrue(os.path.exists("ground_truth.jsonl"))

    def test_evaluate_retrieval(self):
        path1 = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/candidate_context.jsonl"
        if os.path.exists(path1):
            index_file_jsonl_path=path1
        else:
            index_file_jsonl_path='../tools/evaluation/data_augmentation/candidate_context.jsonl'
        path2 = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/example.jsonl"
        if os.path.exists(path2):
            query_file_jsonl_path=path2
        else:
            query_file_jsonl_path='../tools/evaluation/data_augmentation/example.jsonl'
        argv = ['--index_file_jsonl_path', index_file_jsonl_path, \
                '--query_file_jsonl_path', query_file_jsonl_path, \
                '--embedding_model', '/tf_dataset2/inc-ut/gte-base']
        with patch('sys.argv', ['python evaluate_retrieval.py'] + argv):
            result = evaluate_retrieval.main()
            self.assertIsNotNone(result)

    def test_ragas_evaluation(self):
        path1 = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/answer.jsonl"
        if os.path.exists(path1):
            answer_file_path=path1
        else:
            answer_file_path='../tools/evaluation/data_augmentation/answer.jsonl'
        path2 = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/evaluation/data_augmentation/ground_truth.jsonl"
        if os.path.exists(path2):
            ground_truth_file_path=path2
        else:
            ground_truth_file_path='../tools/evaluation/data_augmentation/ground_truth.jsonl'
        argv = ['--answer_file', answer_file_path, \
                '--ground_truth_file', ground_truth_file_path, \
                '--llm_model', '/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3-1', \
                '--embedding_model', '/tf_dataset2/inc-ut/gte-base']
        with patch('sys.argv', ['python ragas_evaluation.py'] + argv):
            result = ragas_evaluation.main()
            self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
