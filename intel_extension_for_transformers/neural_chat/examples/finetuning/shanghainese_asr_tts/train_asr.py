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

from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel
import os, random, time
import torch
import numpy as np
import sys

from dataclasses import dataclass, field

@dataclass
class CustomTrainingArguments(TrainingArguments):
    save_steps: int = field(default=100)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
setup_seed(42)

model_path = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpeechRecognitionModel(model_path=model_path, device=device)
now = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
output_dir = f'./checkpoint-{model_path}-{now}'

train_data = []

def read_data(dataset_name, wav_split):
    data_len = len(os.listdir(f'{dataset_name}/Split_TXT'))
    print(data_len)
    for cnt in range(1, data_len+1):
        transcription = open(f'{dataset_name}/Split_TXT/{cnt}.txt', encoding='utf-8').readline().strip()
        path = f'{dataset_name}/Split_WAV{wav_split}/{cnt}.wav'
        res = model.processor.tokenizer(transcription)['input_ids']
        if all(x == 3 for x in res):
            continue
        train_data.append(
            {"path": path, "transcription":transcription}
        )


# for debug
# read_data('Shanghai_Dialect_Dict', 1)
# read_data('Shanghai_Dialect_Dict', 2)
# random.shuffle(train_data)
# eval_ratio = 0.05
# index = int(len(train_data) * eval_ratio)
# eval_data = train_data[:10]
# train_data = train_data[10:20]
# batch_size = 1
# eval_steps = 100
# fp16 = False

# for train
read_data('Shanghai_Dialect_Conversational_Speech_Corpus', 1)
read_data('Shanghai_Dialect_Conversational_Speech_Corpus', 2)
read_data('Shanghai_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence', 1)
read_data('Shanghai_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence', 2)
read_data('Shanghai_Dialect_Dict', 1)
read_data('Shanghai_Dialect_Dict', 2)
read_data('Shanghai_Dialect_Zhongguoyuyan', 1)

eval_ratio = 0.05
index = int(len(train_data) * eval_ratio)
eval_data = train_data[:index]
train_data = train_data[index:]
batch_size = 32
eval_steps = 100
fp16 = False ##### Disable here to avoid torch 2.x incompatible with huggingsound, which uses accelerate and cause clip gradient bugs with torch.cuda.amp.grad_scaler.GradScaler


print('eval_data_len:', len(eval_data))
print('train_data_len:', len(train_data))

training_args = CustomTrainingArguments(
    save_steps=eval_steps,
    group_by_length=True,
    num_train_epochs=200,
    learning_rate=1e-4,
    eval_steps=eval_steps,
    save_total_limit=2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    early_stopping_patience=5,
    metric_for_best_model='cer',
    fp16=fp16,
    show_dataset_stats=False, # here to avoid long time traversing
    max_grad_norm=None
)
model_args = ModelArguments(
    activation_dropout=0.1,
    hidden_dropout=0.1,
)

model.finetune(
    output_dir, 
    train_data=train_data, 
    eval_data=eval_data,
    training_args=training_args,
    model_args=model_args,
)
