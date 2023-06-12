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

import time
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from transformers import EvalPrediction
import sys
import os
import copy

common_dir = os.path.join(sys.path[0], "../../../../neural_engine_utils/")
sys.path.append(common_dir)
from utils_qa import postprocess_qa_predictions
from executor_dataloader import DataLoader
from common import (log, DummyDataLoader, compute_performance, Neural_Engine_base)


class Neural_Engine(Neural_Engine_base):

    def accuracy(self, batch_size, max_eval_samples, dataset_name, data_dir, tokenizer_dir):
        # load dataset
        log.info("Load dataset ......")
        dataset = DataLoader(batch_size, max_eval_samples, dataset_name, data_dir, tokenizer_dir)
        # load metric
        log.info("Load metric ......")
        metric = load_metric(dataset_name)

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        # execute
        log.info("Start engine ......")
        start_logits_list = []
        end_logits_list = []
        for idx in tqdm(range(len(dataset))):
            inputs = dataset[idx]
            predictions = copy.deepcopy(self.graph.inference(inputs))
            predictions = list(predictions.values())[0]
            start_logits_list.append(predictions[..., 0])
            end_logits_list.append(predictions[..., 1])
        start_logits = np.concatenate(start_logits_list, axis=0)
        end_logits = np.concatenate(end_logits_list, axis=0)
        results = (start_logits, end_logits)
        # post process
        log.info("Post process ......")

        def post_processing_function(examples,
                                     features,
                                     predictions,
                                     answer_column_name,
                                     stage="eval"):
            # Post-processing: we match the start logits and end logits to answers in the original context.
            predictions = postprocess_qa_predictions(examples=examples,
                                                     features=features,
                                                     predictions=predictions,
                                                     max_answer_length=384,
                                                     prefix=stage)
            # Format the result to the format the metric expects.
            formatted_predictions = [{
                "id": k,
                "prediction_text": v
            } for k, v in predictions.items()]
            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        eval_dataset, eval_examples, answer_column_name = dataset.get_eval()
        eval_preds = post_processing_function(eval_examples, eval_dataset, results,
                                              answer_column_name)
        # compute metrics
        log.info("Compute metrics ......")
        eval_metric = compute_metrics(eval_preds)
        # for question answering task, F1 is the most important metric.
        f1_value = eval_metric.get("f1")
        exact_match = eval_metric.get("exact_match")
        log.info(f"F1 Accuracy: {f1_value}")
        log.info(f"Exact Match: {exact_match}")

    def performance(self, batch_size, seq_len, iteration, warm_up):
        if warm_up >= iteration:
            log.error("Warm up should less than iteration.")
            raise ValueError()
        # generate dummy dataset
        log.info("Generate dummy dataset ......")
        shape = [batch_size, seq_len]
        dataset = DummyDataLoader(shapes=[shape, shape, shape],
                                  lows=[1, 1, 1],
                                  highs=[128, 1, 1],
                                  dtypes=['int32', 'int32', 'int32'],
                                  iteration=iteration)
        compute_performance(dataset, self.graph, log, self.log_file, warm_up, batch_size, seq_len)
