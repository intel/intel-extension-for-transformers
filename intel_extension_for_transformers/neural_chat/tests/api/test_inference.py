#!/usr/bin/env python
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

import os
import unittest
from intel_extension_for_transformers.neural_chat.chatbot import build_chatbot, finetune_model, optimize_model
from intel_extension_for_transformers.neural_chat.config import (
    PipelineConfig, GenerationConfig, FinetuningConfig, OptimizationConfig,
    ModelArguments, DataArguments, TrainingArguments, FinetuningArguments
)

class UnitTest(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_text_chat(self):
        config = PipelineConfig(model_name_or_path='./Llama-2-7b-chat-hf')
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_retrieval(self):
        config = PipelineConfig(model_name_or_path='./Llama-2-7b-chat-hf', retrieval_type="sparse", retrieval_document_path="../../assets/docs/")
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_voice_chat(self):
        config = PipelineConfig(model_name_or_path='./Llama-2-7b-chat-hf', audio_output=True)
        chatbot = build_chatbot(config)
        gen_config = GenerationConfig(max_new_tokens=64, audio_output_path='./response.wav')
        response = chatbot.predict(query="Nice to meet you!", config=gen_config)
        print(response)
        self.assertIsNotNone(response)
        print("output audio path: ", response)
        self.assertTrue(os.path.exists(gen_config.audio_output_path))

    def test_finetuning(self):
        model_args = ModelArguments(model_name_or_path='./Llama-2-7b-chat-hf', use_fast_tokenizer=False)
        data_args = DataArguments(train_file='./alpaca_data.json', dataset_concatenation=True)
        training_args = TrainingArguments(gradient_accumulation_steps=1,
                                          do_train=True, learning_rate=1e-4, num_train_epochs=1,
                                          logging_steps=100, save_total_limit=2, overwrite_output_dir=True,
                                          log_level='info', save_strategy='epoch', max_steps=3,
                                          output_dir='./saved_model', no_cuda=True)
        finetune_args = FinetuningArguments(peft='lora')
        config = FinetuningConfig(model_args, data_args, training_args, finetune_args)
        finetune_model(config)


    def test_quantization(self):
        config = OptimizationConfig()
        optimize_model(config)

    


if __name__ == '__main__':
    unittest.main()
