# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import Dataset
import os
from ragas import evaluate   # pylint: disable=E0401
from ragas.metrics import (    # pylint: disable=E0401
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
import pandas as pd
import jsonlines
import argparse


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 10)

def load_set(file_jsonl_path, item):
    list = []
    with open(file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            passages=stu[item]
            list.append(passages)
    return list

def ragas(answer_file, ground_truth_file, openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    question_list=load_set(answer_file, "question")
    answer_list=load_set(answer_file, "answer")
    contexts_list=load_set(ground_truth_file, "context")
    ground_truth_list=load_set(ground_truth_file, "ground_truth")

    data_samples = {
        'question': question_list,
        'answer': answer_list,
        'contexts' : contexts_list,
        'ground_truth': ground_truth_list
    }

    dataset = Dataset.from_dict(data_samples)

    score = evaluate(dataset,metrics=[answer_relevancy, faithfulness, context_recall, context_precision])
    df=score.to_pandas()
    print(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", type=str)
    parser.add_argument("--ground_truth_file", type=str)
    parser.add_argument("--openai_api_key", type=str)
    args = parser.parse_args()

    answer_file = args.answer_file
    ground_truth_file = args.ground_truth_file
    openai_api_key = args.openai_api_key

    ragas(answer_file, ground_truth_file, openai_api_key)

if __name__ == '__main__':
    main()
