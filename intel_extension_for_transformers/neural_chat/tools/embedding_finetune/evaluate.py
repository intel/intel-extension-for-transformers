#!/usr/bin/env python
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

import argparse
import faiss
from faiss import normalize_L2
from sentence_transformers import SentenceTransformer
import jsonlines
import numpy as np

def faiss_retrieval(xb,xq,k):
    d=xb.shape[1]
    normalize_L2(xb)
    normalize_L2(xq)
    nlist=1
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(xb) # pylint: disable=E1120
    index.add(xb) # pylint: disable=E1120

    D, I = index.search(xq, k) # pylint: disable=E1120
    return I

def index_library(index_file_jsonl_path, model_name):
    result_list = []
    with open(index_file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            passages=stu["context"]
            model = SentenceTransformer(model_name)
            p_embeddings = model.encode(passages, normalize_embeddings=True)
            arr = np.array(p_embeddings)
            result_list.append(arr)
            index_arr = np.concatenate(result_list)
    return index_arr

def query_set(query_file_jsonl_path, model_name):
    result_list = []
    with open(query_file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            queries = [stu["query"]]
            model = SentenceTransformer(model_name)
            instruction = ""
            q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
            arr = np.array(q_embeddings)
            result_list.append(arr)
            query_arr = np.concatenate(result_list)
    return query_arr

def load_list(file_jsonl_path, item):
    with open(file_jsonl_path) as file:
        data = []
        for stu in jsonlines.Reader(file):
            content = ",".join(stu[item])
            data.append(content)
    return data


def evaluate(preds, labels, cutoffs=[1]):
    """
    Evaluate MRR and Hit at cutoffs.
    """
    metrics = {}

    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Hit
    hit_rate_list=[]
    for cutoff in cutoffs:
        hit_num = 0
        for pred, label in zip(preds, labels):
            hit_list=np.intersect1d(label, pred[:cutoff])
            hit_num = hit_num+len(hit_list)
        hit_rate = hit_num/len(labels)
        hit_rate_list.append(hit_rate)
    for i, cutoff in enumerate(cutoffs):
        hit_rate = hit_rate_list[i]
        metrics[f"Hit@{cutoff}"] = hit_rate

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--index_file_jsonl_path", type=str)
    parser.add_argument("--query_file_jsonl_path", type=str)
    args = parser.parse_args()

    model_name = args.model_name
    index_file_jsonl_path = args.index_file_jsonl_path
    query_file_jsonl_path = args.query_file_jsonl_path

    index_arr = index_library(index_file_jsonl_path, model_name)
    query_arr = query_set(query_file_jsonl_path, model_name)

    indices = faiss_retrieval(index_arr,query_arr,1)

    retrieval_results = []
    for indice in indices:
        indice = indice[indice != -1].tolist()
        corpus_pos=load_list(index_file_jsonl_path, "context")
        results = []
        for i in indice:
            results.append(corpus_pos[i])
        retrieval_results.append(results)

    ground_truths=load_list(query_file_jsonl_path, "pos")

    metrics = evaluate(retrieval_results, ground_truths)
    print(metrics)
    return metrics

if __name__ == '__main__':
    main()
