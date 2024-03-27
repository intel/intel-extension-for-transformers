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
import os, shutil
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
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import plugins
from intel_extension_for_transformers.neural_chat.config import GenerationConfig

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

def ragas(answer_file, ground_truth_file):

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

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    score = evaluate(dataset,metrics=[answer_relevancy, faithfulness, context_recall, context_precision])

    df=score.to_pandas()
    answer_relevancy_average=df['answer_relevancy'][:].mean()
    faithfulness_average=df['faithfulness'][:].mean()
    context_recall_average=df['context_recall'][:].mean()
    context_precision_average=df['context_precision'][:].mean()
    return answer_relevancy_average, faithfulness_average, context_recall_average, context_precision_average


def rag(text,
        input_path,
        vector_database="Chroma",
        embedding_model="BAAI/bge-large-en-v1.5",
        retrieval_type='default',
        max_chuck_size=256,
        search_type="similarity",
        k=1,
        fetch_k=5,
        score_threshold=0.3,
        polish=False,
        top_n=1,
        enable_rerank=False,
        reranker_model="BAAI/bge-reranker-large",
        llm_model='intel/neural-chat-7b-v3-1',
        temperature=0.01,
        top_k=1,
        top_p=0.1,
        repetition_penalty=1.0,
        num_beams=1,
        do_sample=True
        ):
    plugins.retrieval.enable=True
    plugins.retrieval.args["input_path"]=input_path
    plugins.retrieval.args["vector_database"]=vector_database
    plugins.retrieval.args["embedding_model"]=embedding_model
    plugins.retrieval.args["retrieval_type"]=retrieval_type
    plugins.retrieval.args["max_chuck_size"]=max_chuck_size
    plugins.retrieval.args["search_type"]=search_type
    if search_type=="similarity":
        plugins.retrieval.args["search_kwargs"]={"k":k}
    elif search_type=="mmr":
        plugins.retrieval.args["search_kwargs"]={"k":k, "fetch_k":fetch_k}
    elif search_type=="similarity_score_threshold":
        plugins.retrieval.args["search_kwargs"]={"k":k, "score_threshold":score_threshold}
    plugins.retrieval.args["polish"]=polish
    plugins.retrieval.args["top_n"]=top_n
    plugins.retrieval.args["enable_rerank"]=enable_rerank
    plugins.retrieval.args["reranker_model"]=reranker_model
    config = PipelineConfig(plugins=plugins, model_name_or_path=llm_model)
    chatbot = build_chatbot(config)
    response = chatbot.predict(text,
                            config=GenerationConfig(temperature=temperature,
                                                    top_k=top_k,
                                                    top_p=top_p,
                                                    repetition_penalty=repetition_penalty,
                                                    num_beams=num_beams,
                                                    do_sample=do_sample))
    return response

def result_data(ground_truth_file,
                input_path,
                vector_database="Chroma",
                embedding_model="BAAI/bge-large-en-v1.5",
                retrieval_type='default',
                max_chuck_size=256,
                search_type="similarity",
                k=1,
                fetch_k=5,
                score_threshold=0.3,
                polish=False,
                top_n=1,
                enable_rerank=False,
                reranker_model="BAAI/bge-reranker-large",
                llm_model='intel/neural-chat-7b-v3-1',
                temperature=0.01,
                top_k=1,
                top_p=0.1,
                repetition_penalty=1.0,
                num_beams=1,
                do_sample=True
                ):
    question_list = load_set(ground_truth_file, "question")

    result_answer_path='result_answer.jsonl'
    if os.path.exists("result_answer.jsonl"):
        os.remove("result_answer.jsonl")

    if os.path.exists("output"):
        shutil.rmtree("output", ignore_errors=True)
    for question in question_list:
        response = rag(
                        question,
                        input_path,
                        vector_database,
                        embedding_model,
                        retrieval_type,
                        max_chuck_size,
                        search_type,
                        k,
                        fetch_k,
                        score_threshold,
                        polish,
                        top_n,
                        enable_rerank,
                        reranker_model,
                        llm_model,
                        temperature,
                        top_k,
                        top_p,
                        repetition_penalty,
                        num_beams,
                        do_sample
                     )
        data = {
                "question": question,
                "answer": response,
            }
        with jsonlines.open(result_answer_path,"a") as file_json:
                file_json.write(data)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ground_truth_file", type=str)
    parser.add_argument("--input_path", type=str)

    parser.add_argument("--vector_database", type=str, default="Chroma")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--llm_model", type=str)
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-large")

    parser.add_argument("--retrieval_type", type=str, default='default')
    parser.add_argument("--polish", type=bool, default=False)
    parser.add_argument("--search_type", type=str, default="similarity")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--fetch_k", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--enable_rerank", type=bool, default=False)

    parser.add_argument("--max_chuck_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=True)

    args = parser.parse_args()

    ground_truth_file = args.ground_truth_file
    input_path = args.input_path
    vector_database = args.vector_database
    embedding_model = args.embedding_model
    retrieval_type = args.retrieval_type
    polish = args.polish
    search_type = args.search_type
    llm_model = args.llm_model
    k = args.k
    fetch_k = args.fetch_k
    score_threshold = args.score_threshold
    reranker_model = args.reranker_model
    top_n = args.top_n
    enable_rerank = args.enable_rerank

    max_chuck_size = args.max_chuck_size
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty
    num_beams = args.num_beams
    do_sample = args.do_sample

    result_data(ground_truth_file,
                input_path,
                vector_database=vector_database,
                embedding_model=embedding_model,
                retrieval_type=retrieval_type,
                max_chuck_size=max_chuck_size,
                search_type=search_type,
                k=k,
                fetch_k=fetch_k,
                score_threshold=score_threshold,
                polish=polish,
                top_n=top_n,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                llm_model=llm_model,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                do_sample=do_sample)

    answer_file = 'result_answer.jsonl'
    answer_relevancy_average,faithfulness_average,context_recall_average,context_precision_average=ragas(
        answer_file,
        ground_truth_file)

    file_json_path='result_ragas.jsonl'

    if answer_relevancy_average and faithfulness_average and context_recall_average and context_precision_average:
        data = {
                "ground_truth_file": args.ground_truth_file,
                "input_path": args.input_path,
                "vector_database": args.vector_database,
                "embedding_model": args.embedding_model,
                "retrieval_type": args.retrieval_type,
                "polish": args.polish,
                "search_type": args.search_type,
                "llm_model": args.llm_model,
                "k": args.k,
                "fetch_k": args.fetch_k,
                "score_threshold": args.score_threshold,
                "reranker_model": args.reranker_model,
                "top_n": args.top_n,
                "enable_rerank": args.enable_rerank,
                "max_chuck_size": args.max_chuck_size,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "num_beams": args.num_beams,
                "do_sample": args.do_sample,
                "answer_relevancy_average": answer_relevancy_average,
                "faithfulness_average": faithfulness_average,
                "context_recall_average": context_recall_average,
                "context_precision_average": context_precision_average,
            }
        print(data)
        with jsonlines.open(file_json_path,"a") as file_json:
                file_json.write(data)

if __name__ == '__main__':
    main()
