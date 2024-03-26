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

import unittest
import shutil
import os
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model
from intel_extension_for_transformers.neural_chat.config_logging import configure_logging
from intel_extension_for_transformers.neural_chat.utils.common import is_hpu_available, get_device_type
logger = configure_logging()

json_data = \
"""
[
    {"instruction": "Generate a slogan for a software company", "input": "", "output": "The Future of Software is Here"},
    {"instruction": "Provide the word that comes immediately after the.", "input": "He threw the ball over the fence.", "output": "fence."}
]
"""
test_data_file = './alpaca_test.json'

class TestFinetuning(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = get_device_type()
        with open(test_data_file, mode='w') as f:
            f.write(json_data)

        if is_hpu_available:
            from optimum.habana import GaudiTrainingArguments, GaudiSeq2SeqTrainingArguments
            self.training_args = GaudiTrainingArguments(
                    output_dir='./tmp',
                    do_train=True,
                    max_steps=3,
                    overwrite_output_dir=True,
                    use_habana=True,
                    use_lazy_mode=True,
                    bf16=True)

            self.seq2seq_training_args = GaudiSeq2SeqTrainingArguments(
                    output_dir='./tmp',
                    do_train=True,
                    max_steps=3,
                    overwrite_output_dir=True,
                    use_habana=True,
                    use_lazy_mode=True,
                    bf16=True)
        else:
            self.training_args = TrainingArguments(
                    output_dir='./tmp',
                    do_train=True,
                    max_steps=3,
                    overwrite_output_dir=True)

            self.seq2seq_training_args = Seq2SeqTrainingArguments(
                    output_dir='./tmp',
                    do_train=True,
                    max_steps=3,
                    overwrite_output_dir=True)


    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp', ignore_errors=True)
        os.remove(test_data_file)

    def test_finetune_clm_lora(self):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        finetune_model(finetune_cfg)

    def test_finetune_clm_qlora(self):
        if self.device == "hpu":
            logger.info("hpu pass")
            return

        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, qlora=True, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args,
        )
        finetune_model(finetune_cfg)

    def test_finetune_clm_full_finetuning(self):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, full_finetune=True,
                bits=16, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args,
        )
        finetune_model(finetune_cfg)

    def test_finetune_clm_value1(self):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, full_finetune=True, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args,
        )
        # finetune_model(finetune_cfg)
        try:
            finetune_model(finetune_cfg)
        except ValueError:
            logger.info("code pass")

    def test_finetune_clm_value2(self):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, qlora=True,
                full_finetune=True, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args,
        )
        try:
            finetune_model(finetune_cfg)
        except ValueError:
            logger.info("code pass")

    def test_finetune_seq2seq_lora(self):
        model_args = ModelArguments(model_name_or_path="google/flan-t5-small")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.seq2seq_training_args,
            finetune_args=finetune_args,
        )
        finetune_model(finetune_cfg)

    def test_finetune_seq2seq_qlora(self):
        if self.device == "hpu":
            logger.info("hpu pass")
            return

        model_args = ModelArguments(model_name_or_path="google/flan-t5-small")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, qlora=True)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.seq2seq_training_args,
            finetune_args=finetune_args,
        )
        finetune_model(finetune_cfg)

    def test_finetune_seq2seq_full_finetuning(self):
        model_args = ModelArguments(model_name_or_path="google/flan-t5-small")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, full_finetune=True,
                bits=16, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.seq2seq_training_args,
            finetune_args=finetune_args,
        )
        finetune_model(finetune_cfg)

if __name__ == "__main__":
    unittest.main()
