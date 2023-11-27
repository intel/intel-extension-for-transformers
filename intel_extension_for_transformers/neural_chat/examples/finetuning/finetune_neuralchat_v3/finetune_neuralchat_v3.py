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

from optimum.habana import GaudiTrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
data_path = "Open-Orca/SlimOrca"
model_name_or_path = "mistralai/Mistral-7B-v0.1"
model_args = ModelArguments(
    model_name_or_path=model_name_or_path,
    use_fast_tokenizer=False,
)
data_args = DataArguments(
    dataset_name=data_path,
    max_seq_length=2560,
    max_source_length=1024,
    preprocessing_num_workers=4,
    validation_split_percentage=0,
)
training_args = GaudiTrainingArguments(
    output_dir="./finetuned_model",
    overwrite_output_dir=True,
    do_train=True, do_eval=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4, num_train_epochs=2,
    lr_scheduler_type="cosine", warmup_ratio=0.03,
    weight_decay=0.0, save_strategy="steps",
    save_steps=1000, log_level="info", logging_steps=10,
    save_total_limit=10, bf16=True, use_habana=True,
    use_lazy_mode=True,
    report_to=None,
    deepspeed="ds_config.json"
)
finetune_args = FinetuningArguments(
    lora_alpha=64, lora_rank=16, lora_dropout=0.05,
    lora_target_modules=['k_proj', 'q_proj', 'o_proj', 'v_proj'],
    lora_all_linear=False, do_lm_eval=False,
    task="SlimOrca", device="hpu"
)
finetune_cfg = TextGenerationFinetuningConfig(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetune_args=finetune_args,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model
finetune_model(finetune_cfg)
