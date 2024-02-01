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

import argparse
from executor_utils import log, Neural_Engine
from engine_model import EngineBGEModel
from mteb import MTEB


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        default="./model_and_tokenizer/int8-model.onnx",
                        type=str,
                        help="Input model path.")
    parser.add_argument("--mode",
                        default="accuracy",
                        type=str,
                        help="Benchmark mode of performance or accuracy.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--seq_len", default=128, type=int, help="Sequence length.")
    parser.add_argument("--warm_up",
                        default=5,
                        type=int,
                        help="Warm up iteration in performance mode.")
    parser.add_argument("--iteration", default=10, type=int, help="Iteration in performance mode.")
    parser.add_argument("--tokenizer_dir",
                        default="textattack/bert-base-uncased-MRPC",
                        type=str,
                        help="Pre-trained model tokenizer name or path")
    parser.add_argument("--data_dir", default="./data", type=str, help="Data cache directory.")
    parser.add_argument("--dataset_name", default="glue", type=str, help="Name of dataset.")
    parser.add_argument("--task_name", default="mrpc", type=str, help="Task name of dataset.")
    parser.add_argument("--log_file",
                        default="executor.log",
                        type=str,
                        help="File path to log information.")
    parser.add_argument("--dynamic_quantize",
                        default=False,
                        type=bool,
                        help="dynamic quantize for fp32 model.")


    parser.add_argument('--model_name_or_path', default=None, type=str)
    parser.add_argument('--task_type', default=None, type=str, help="task type. Default is None, which means using all task types")
    parser.add_argument('--task_names', default=None, type=str, )
    parser.add_argument('--add_instruction', action='store_true', help="whether to add instruction for query")
    parser.add_argument('--pooling_method', default='cls', type=str)
    parser.add_argument('--ort_model_path', default=None, type=str)
    parser.add_argument('--file_name', default=None, type=str)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.dynamic_quantize:
        executor = Neural_Engine(args.input_model, args.log_file, "dynamic_int8")
    else:
        executor = Neural_Engine(args.input_model, args.log_file, "native")
    if args.mode == "accuracy":
        query_instruction_for_retrieval_dict = {
            "BAAI/bge-large-en": "Represent this sentence for searching relevant passages: ",
            "BAAI/bge-base-en": "Represent this sentence for searching relevant passages: ",
            "BAAI/bge-small-en": "Represent this sentence for searching relevant passages: ",
            "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
            "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
            "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
        }

        model = EngineBGEModel(model_name_or_path=args.model_name_or_path,
                        normalize_embeddings=False,  # normalize embedding will harm the performance of classification task
                        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                        ort_model_path=args.ort_model_path,
                        file_name=args.file_name,
                        engine_model=executor,
                        backend='Engine')

        if args.task_names is None:
            task_names = [t.description["name"] for t in MTEB(task_types=args.task_type,
                                                            task_langs=['en']).tasks]
        else:
            task_names = [args.task_names]
        print("task_names", task_names)
        
        results = {}
        for task in task_names:
            model.query_instruction_for_retrieval = None

            evaluation = MTEB(tasks=[task], task_langs=['en'], eval_splits = ["test" if task not in ['MSMARCO'] else 'dev'])
            result = evaluation.run(model, output_folder=f"en_results/{args.model_name_or_path.split('/')[-1]}")
            results.update(result)
            print(results)
        avg_res = 0
        
        for task_name, task_res in results.items():
            if task_name in ['STS17']:
                avg_res += round(task_res['test']['en-en']['cos_sim']['spearman'] * 100, 2)
            elif task_name in ['STS22']:
                avg_res += round(task_res['test']['en']['cos_sim']['spearman'] * 100, 2)
            else:
                avg_res += round(task_res['test']['cos_sim']['spearman'] * 100, 2)
        avg_res /= len(results)
        print("STS res", avg_res)

    elif args.mode == "performance":
        executor.performance(args.batch_size, args.seq_len, args.iteration, args.warm_up)
    else:
        log.error("Benchmark only has performance or accuracy mode")
