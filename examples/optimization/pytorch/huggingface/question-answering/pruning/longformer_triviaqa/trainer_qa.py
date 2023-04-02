# coding=utf-8

# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
"""
This script is based on HuggingFace/transformers example: https://github.com/huggingface/transformers/blob/v4.6.1/examples/pytorch/question-answering/trainer_qa.py
"""

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
import utils_qa
import collections
from collections import defaultdict
import numpy as np
import torch
import json


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = utils_qa.evaluate_triviaqa(eval_preds.label_ids, eval_preds.predictions)
            #metrics = self.compute_metrics(eval_preds)

            #self.log(metrics)
        else:
            metrics = {}

        #if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #    xm.master_print(met.metrics_report())

        #self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, n_best_size=20, max_answer_length=30):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            predict_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )

        all_start_logits, all_end_logits = output.predictions

        all_predictions = collections.OrderedDict()

        qa_with_duplicates = defaultdict(list)

        for example_index, example in enumerate(predict_examples):
            input_ids = torch.tensor([predict_dataset[example_index]["input_ids"]])
            qid = predict_dataset[example_index]["example_id"]

            eos_token_indices = (input_ids == self.tokenizer.eos_token_id).nonzero()
            question_end_index = eos_token_indices.view(input_ids.size(0), 2, 2)[:, 0, 1]
            start_logits = all_start_logits[example_index]
            end_logits = all_end_logits[example_index]
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            potential_answers = []
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index <= question_end_index[0]:
                        continue
                    if end_index <= question_end_index[0]:
                        continue
                    if start_index > end_index:
                        continue
                    answer_len = end_index - start_index + 1
                    if answer_len > max_answer_length:
                        continue
                    potential_answers.append({'start': start_index, 'end': end_index,
                        'start_logit': start_logits[start_index].item(),
                        'end_logit': end_logits[end_index].item()})
            sorted_answers = sorted(potential_answers, key=lambda x: (x['start_logit'] + x['end_logit']), reverse=True)
            if len(sorted_answers) == 0:
                answer = {'text': 'NoAnswerFound', 'score': -1000000}
            else:
                answer = sorted_answers[0]
                answer_token_ids = input_ids[0, answer['start']: answer['end'] + 1]
                answer_tokens = self.tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())
                text = self.tokenizer.convert_tokens_to_string(answer_tokens)
                score = answer['start_logit'] + answer['end_logit']
                answer = {'text': text, 'score': score}
            qa_with_duplicates[qid].append({'answer_score': answer['score'], 'answer_text': answer['text'], })

        qid_to_answer_text = {}
        for qid, answer_metrics in qa_with_duplicates.items():
            top_answer = sorted(answer_metrics, key=lambda x: x['answer_score'], reverse=True)[0]
            qid_to_answer_text[qid] = top_answer['answer_text']

        with open('predictions.json', 'w') as f:
            f.write(json.dumps(qid_to_answer_text, indent=4) + "\n")
