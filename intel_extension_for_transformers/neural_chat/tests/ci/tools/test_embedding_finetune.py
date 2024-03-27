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
from intel_extension_for_transformers.neural_chat.tools.embedding_finetune import finetune, mine_hard_neg, evaluate

class TestEmbeddingFinetune(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("BAAI"):
            shutil.rmtree("BAAI", ignore_errors=True)
        if os.path.exists("augmented_example.jsonl"):
            os.remove("augmented_example.jsonl")
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("BAAI"):
            shutil.rmtree("BAAI", ignore_errors=True)
        if os.path.exists("augmented_example.jsonl"):
            os.remove("augmented_example.jsonl")
        return super().tearDown()

    def test_finetune(self):
        path = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/augmented_example.jsonl"
        if os.path.exists(path):
            train_data_path=path
        else:
            train_data_path='../tools/embedding_finetune/augmented_example.jsonl'
        argv = ['--output_dir', 'BAAI/bge-base-en-v1.5_annual', \
                '--model_name_or_path', 'BAAI/bge-base-en-v1.5', \
                '--train_data', train_data_path, \
                '--learning_rate', '1e-5', \
                '--num_train_epochs', '5', \
                '--per_device_train_batch_size', '1', \
                '--dataloader_drop_last', 'True', \
                '--normalized', 'True', \
                '--temperature', '0.02', \
                '--query_max_len', '64', \
                '--passage_max_len', '256', \
                '--train_group_size', '2', \
                '--logging_steps', '10', \
                '--query_instruction_for_retrieval', '""', \
                '--bf16', 'True']

        with patch('sys.argv', ['python finetune.py'] + argv):
            finetune.main()
            self.assertTrue(os.path.exists("BAAI/bge-base-en-v1.5_annual"))

    def test_mine_hard_neg(self):
        path = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/example.jsonl"
        if os.path.exists(path):
            input_file_path=path
        else:
            input_file_path='../tools/embedding_finetune/example.jsonl'
        argv = ['--model_name_or_path', 'BAAI/bge-base-en-v1.5', \
                '--input_file', input_file_path, \
                '--output_file', 'augmented_example.jsonl', \
                '--range_for_sampling', '2-10', \
                '--negative_number', '5']
        with patch('sys.argv', ['python mine_hard_neg.py'] + argv):
            mine_hard_neg.main()
            self.assertTrue(os.path.exists("augmented_example.jsonl"))

    def test_evaluate(self):
        path1 = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/candidate_context.jsonl"
        if os.path.exists(path1):
            index_file_jsonl_path=path1
        else:
            index_file_jsonl_path='../tools/embedding_finetune/candidate_context.jsonl'
        path2 = \
          "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tools/embedding_finetune/example.jsonl"
        if os.path.exists(path2):
            query_file_jsonl_path=path2
        else:
            query_file_jsonl_path='../tools/embedding_finetune/example.jsonl'
        argv = ['--model_name', 'BAAI/bge-base-en-v1.5', \
                '--index_file_jsonl_path', index_file_jsonl_path, \
                '--query_file_jsonl_path', query_file_jsonl_path]
        with patch('sys.argv', ['python evaluate.py'] + argv):
            metrics=evaluate.main()
            self.assertIsNotNone(metrics)

if __name__ == '__main__':
    unittest.main()
