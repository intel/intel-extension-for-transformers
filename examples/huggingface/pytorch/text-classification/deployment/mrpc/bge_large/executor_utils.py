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
from executor_dataloader import DataLoader
import sys
import os
from mteb import MTEB
from tqdm.autonotebook import trange
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np

common_dir = os.path.join(sys.path[0], "../../../../neural_engine_utils/")
sys.path.append(common_dir)
from common import (log, DummyDataLoader, compute_performance, Neural_Engine_base)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

class Neural_Engine(Neural_Engine_base):

    def accuracy(self, batch_size, seq_len, dataset_name, task_name, data_dir, tokenizer_dir):
        # # load dataset
        # log.info("Load dataset ......")
        # dataset = DataLoader(batch_size, seq_len, dataset_name, task_name, data_dir, tokenizer_dir)
        # # load metric
        # log.info("Load metric ......")
        # if dataset_name and task_name is not None:
        #     metric = load_metric(dataset_name, task_name)
        # else:
        #     metric = load_metric("accuracy")
        # # execute
        # log.info("Start engine ......")
        # for idx in tqdm(range(len(dataset))):
        #     inputs = dataset[idx][0]
        #     labels = dataset[idx][1]
        #     predictions = self.graph.inference(inputs)
            
        #     predictions = list(predictions.values())[0]
        #     import pdb;pdb.set_trace()
        #     predictions = np.argmax(predictions, axis=1)
        #     metric.add_batch(
        #         predictions=predictions,
        #         references=labels,
        #     )
        # # compute metrics
        # log.info("Compute metrics ......")
        # eval_metric = metric.compute()
        # accuracy_value = eval_metric.get("accuracy")
        # f1_value = eval_metric.get("f1")
        # log.info(f"Accuracy: {accuracy_value}")
        # log.info(f"F1: {f1_value}")

        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        engine_graph = self.graph
        class MyModel():
            def __init__(
                self,
                engine_graph,
                tokenizer
            ):
                self._graph = engine_graph
                self._tokenizer = tokenizer
                self.flag = 0

            def _text_length(self, text: Union[List[int], List[List[int]]]):
                """
                Help function to get the length for the input text. Text can be either
                a list of ints (which means a single text as input), or a tuple of list of ints
                (representing several text inputs to the model).
                """

                if isinstance(text, dict):              #{key: value} case
                    return len(next(iter(text.values())))
                elif not hasattr(text, '__len__'):      #Object has no len() method
                    return 1
                elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
                    return len(text)
                else:
                    return sum([len(t) for t in text])      #Sum of length of individual strings

            def encode(self, sentences, batch_size=32, output_value: str = 'sentence_embedding', **kwargs):
                """
                Returns a list of embeddings for the given sentences.
                Args:
                    sentences (`List[str]`): List of sentences to encode
                    batch_size (`int`): Batch size for the encoding

                Returns:
                    `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
                """

                length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
                sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

                for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
                    sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                    #features = self._tokenizer(sentences_batch)
                    #out_features = self.forward(features)
                    for sentence in sentences_batch:
                        features = self._tokenizer(sentence)
                        features['input_ids']
                        inputs = []
                        inputs.append(np.asarray(features['input_ids']).reshape(1, -1))
                        inputs.append(np.asarray(features['token_type_ids']).reshape(1, -1))
                        inputs.append(np.asarray(features['attention_mask']).reshape(1, -1))
                        import pdb;pdb.set_trace()
                        out_features = self._graph.inference(inputs)

                    #Sentence embeddings
                    embeddings = out_features[output_value]
                #     embeddings = embeddings.detach()
                #     if normalize_embeddings:
                #         embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                #     # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                #     if convert_to_numpy:
                #         embeddings = embeddings.cpu()

                # all_embeddings.extend(embeddings)


        model = MyModel(self.graph, tokenizer)
        evaluation = MTEB(task_langs=['en'], tasks=['AmazonCounterfactualClassification'])
        results = evaluation.run(model)
        print(results)

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
