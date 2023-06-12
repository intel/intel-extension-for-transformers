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

common_dir = os.path.join(sys.path[0], "../")
sys.path.append(common_dir)
from utils_qa import postprocess_qa_predictions
from executor_dataloader import DataLoader
from common import (log, set_log_file, load_graph, DummyDataLoader)


class IPEX(object):
    def __init__(self, model_path, log_file):
        set_log_file(log, log_file)
        self.graph = load_graph(model_path)
        self.log_file = log_file

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

        import torch
        import intel_extension_for_pytorch as ipex
        model = self.graph
        model = model.to(memory_format=torch.channels_last)
        model = ipex.optimize(model)
        import torch.backends.mkldnn
        torch.backends.quantized.engine = 'onednn'
        for idx in tqdm(range(len(dataset))):
            inputs = {
                'input_ids': torch.from_numpy(dataset[idx][0]),
                'token_type_ids': torch.from_numpy(dataset[idx][1]),
                'attention_mask': torch.from_numpy(dataset[idx][2])
            }
            predictions = model(**inputs)
            start_logits_list.append(predictions['start_logits'].detach().numpy())
            end_logits_list.append(predictions['end_logits'].detach().numpy())
        start_logits = np.concatenate(start_logits_list, axis=0)
        end_logits = np.concatenate(end_logits_list, axis=0)
        results = (start_logits, end_logits)
        # post process
        log.info("Post process ......")

        def post_processing_function(examples, features, predictions, answer_column_name, stage="eval"):
            # Post-processing: we match the start logits and end logits to answers in the original context.
            predictions = postprocess_qa_predictions(examples=examples,
                                                     features=features,
                                                     predictions=predictions,
                                                     max_answer_length=384,
                                                     prefix=stage)
            # Format the result to the format the metric expects.
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        eval_dataset, eval_examples, answer_column_name = dataset.get_eval()
        eval_preds = post_processing_function(eval_examples, eval_dataset, results, answer_column_name)
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

        def compute_performance(dataset, graph, log, log_file, warm_up, batch_size, seq_len):
            log.info("Start executor ......")
            import torch
            import intel_extension_for_pytorch as ipex
            model = graph
            model = model.to(memory_format=torch.channels_last)
            model = ipex.optimize(model)
            import torch.backends.mkldnn
            torch.backends.quantized.engine = 'onednn'
            duration = []
            for idx in tqdm(range(len(dataset))):
                inputs = {
                    'input_ids': torch.from_numpy(dataset[idx][0]),
                    'token_type_ids': torch.from_numpy(dataset[idx][1]),
                    'attention_mask': torch.from_numpy(dataset[idx][2])
                }
                start_time = time.time()
                predictions = model(**inputs)
                end_time = time.time()
                duration.append(end_time - start_time)
            log.info("End executor ......")
            duration_w = duration[warm_up:]
            all_latency = log_file.replace('.log', '.npy')
            _, file_name = os.path.split(all_latency)
            _ = os.getcwd() + '/all_latency'
            try:
                if os.path.exists(_) == False:
                    os.mkdir(_)
            except:
                pass
            all_latency = os.path.join(_, file_name)
            All_latency = np.array(duration_w)
            np.save(all_latency, All_latency, allow_pickle=True, fix_imports=True)
            ave_latency = np.array(duration_w).mean() / batch_size
            p50_latency = np.percentile(duration_w, 50) / batch_size
            p90_latency = np.percentile(duration_w, 90) / batch_size
            p99_latency = np.percentile(duration_w, 99) / batch_size
            log.info("Batch size = {}".format(batch_size))
            log.info("Sequence length = {}".format(seq_len))
            log.info("P50 Latency: {:.3f} ms".format(p50_latency * 1000))
            log.info("P90 Latency: {:.3f} ms".format(p90_latency * 1000))
            log.info("P99 Latency: {:.3f} ms".format(p99_latency * 1000))
            log.info("Average Latency: {:.3f} ms".format(ave_latency * 1000))
            log.info("Throughput: {:.3f} samples/sec".format(1. / ave_latency))
        compute_performance(dataset, self.graph, log, self.log_file, warm_up, batch_size, seq_len)
