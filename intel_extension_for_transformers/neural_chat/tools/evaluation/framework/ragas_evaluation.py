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
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from intel_extension_for_transformers.langchain_community.embeddings import HuggingFaceEmbeddings, \
    HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings  # pylint: disable=E0401, E0611
from langchain_community.embeddings import GooglePalmEmbeddings
from ragas.llms import LangchainLLMWrapper   # pylint: disable=E0611
from ragas.embeddings import LangchainEmbeddingsWrapper   # pylint: disable=E0611
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

def ragas(answer_file, ground_truth_file, llm_model, embedding_model):

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

    if llm_model and embedding_model:
        langchain_llm = HuggingFacePipeline.from_model_id(
            model_id=llm_model,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 128},
        )
        if "instruct" in embedding_model:
            langchain_embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model)
        elif "bge" in embedding_model:
            langchain_embeddings = HuggingFaceBgeEmbeddings(
                model_name=embedding_model,
                encode_kwargs={'normalize_embeddings': True},
                query_instruction="Represent this sentence for searching relevant passages:")
        elif "Google" == embedding_model:
            langchain_embeddings = GooglePalmEmbeddings()
        else:
            langchain_embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )

        langchain_llm = LangchainLLMWrapper(langchain_llm)
        langchain_embedding = LangchainEmbeddingsWrapper(langchain_embeddings)
        score = evaluate(dataset,    # pylint: disable=E1123
                         metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
                         llm = langchain_llm,    # pylint: disable=E1123
                         embeddings=langchain_embedding)    # pylint: disable=E1123
    else:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        score = evaluate(dataset,metrics=[answer_relevancy, faithfulness, context_recall, context_precision])

    df=score.to_pandas()
    print(df)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", type=str)
    parser.add_argument("--ground_truth_file", type=str)
    parser.add_argument("--llm_model", type=str)
    parser.add_argument("--embedding_model", type=str)
    args = parser.parse_args()

    answer_file = args.answer_file
    ground_truth_file = args.ground_truth_file
    llm_model = args.llm_model
    embedding_model = args.embedding_model

    metrics=ragas(answer_file, ground_truth_file, llm_model, embedding_model)
    return metrics

if __name__ == '__main__':
    main()
