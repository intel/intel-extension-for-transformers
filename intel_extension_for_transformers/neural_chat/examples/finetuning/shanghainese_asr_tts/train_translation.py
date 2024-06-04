# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import os, time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

train_data = {}
sh_list = []
zh_list = []
def read_data(dataset_name):
    data_len = len(os.listdir(f'{dataset_name}/Split_TXT'))
    print(data_len)
    for cnt in range(1, data_len+1):
        sh = open(f'{dataset_name}/Split_TXT/{cnt}.txt', encoding='utf-8').readline().strip()
        zh = open(f'{dataset_name}/Split_PROMPT/{cnt}.txt', encoding='utf-8').readline().strip()
        sh_list.append(sh)
        zh_list.append(zh)

read_data('Shanghai_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence')
read_data('Shanghai_Dialect_Dict')
read_data('Shanghai_Dialect_Ximalaya')
read_data('Shanghai_Dialect_Zhongguoyuyan')

data = {'train':Dataset.from_dict({'sh':sh_list,'zh':zh_list})}
data = DatasetDict(data)
data = data["train"].train_test_split(test_size=0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(device)

now = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
output_dir = f'./checkpoint-opus-mt-en-zh-{now}'

def preprocess_function(examples):
    inputs = examples['sh']
    targets = examples['zh']

    with tokenizer.as_target_tokenizer():
        model_inputs = tokenizer(inputs, max_length=64, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=64, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

batch_size = 32

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=200,
    eval_steps=100,
    load_best_model_at_end=True,
)
callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
)

trainer.train()
