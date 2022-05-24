#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import math
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

class DataLoader(object):
    def __init__(self, batch_size, max_eval_samples, dataset_name, data_dir, tokenizer_dir):
        self.batch_size = batch_size
        self.eval_examples = load_dataset(dataset_name, None, cache_dir=data_dir, split='validation')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        column_names = self.eval_examples.column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        self.answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"

        # Validation preprocessing
        def prepare_validation_features(examples):
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=384,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples

        if max_eval_samples != -1:
            # We will select sample from whole data
            max_eval_samples = min(len(self.eval_examples), max_eval_samples)
            self.eval_examples = self.eval_examples.select(range(max_eval_samples))

        # Validation Feature Creation
        self.eval_dataset = self.eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on validation eval_dataset",
        )

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        if end > len(self.eval_dataset):
            input_ids_data = self.eval_dataset[start:]['input_ids']
            segment_ids_data = self.eval_dataset[start:]['token_type_ids']
            input_mask_data = self.eval_dataset[start:]['attention_mask']
        else:
            input_ids_data = self.eval_dataset[start:end]['input_ids']
            segment_ids_data = self.eval_dataset[start:end]['token_type_ids']
            input_mask_data = self.eval_dataset[start:end]['attention_mask']

        sample_size = len(input_ids_data) if isinstance(input_ids_data, list) else 1

        return [np.array(input_ids_data).reshape(sample_size, -1).astype('int32'),
                np.array(segment_ids_data).reshape(sample_size, -1).astype('int32'),
                np.array(input_mask_data).reshape(sample_size, -1).astype('int32')]

    def __len__(self):
        return math.ceil(len(self.eval_dataset)/self.batch_size)

    def get_eval(self):
        return self.eval_dataset, self.eval_examples, self.answer_column_name
