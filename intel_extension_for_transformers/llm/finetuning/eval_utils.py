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

import evaluate
import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def compute_rouge_metric(model, tokenizer, eval_dataset, training_args, gen_kwargs):
    model.eval()
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # Metric
    metric = evaluate.load("rouge")

    def collate_fn(batch):
        input_ids = [torch.tensor(ins["decoder_input_ids"]) for ins in batch]
        labels = [torch.tensor(ins["decoder_labels"]) for ins in batch]
        attention_mask = [torch.tensor(ins["decoder_attention_mask"]) for ins in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                )

    # TODO: support batch_size >1
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn,
                            batch_size=1)


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    for step, batch in enumerate(eval_dataloader):
        preds = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                **gen_kwargs,
                )
        labels = batch["labels"]
        labels = labels.cpu().numpy()

        preds = preds.cpu().numpy()

        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id).tolist()
        # only pred
        preds = [pred[batch["input_ids"].shape[1]:] for pred in preds]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).tolist()
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
                )


    result = metric.compute(use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result
