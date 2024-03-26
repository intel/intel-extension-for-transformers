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

from typing import ClassVar, Collection
from intel_extension_for_transformers.langchain_community.embeddings import HuggingFaceEmbeddings, \
    HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings  # pylint: disable=E0401, E0611
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from intel_extension_for_transformers.langchain_community.vectorstores import Chroma, Qdrant  # pylint: disable=E0401, E0611
import uuid
from langchain_core.documents import Document
from intel_extension_for_transformers.langchain_community.retrievers import ChildParentRetriever  # pylint: disable=E0401, E0611
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.detector.query_explainer \
    import QueryPolisher
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
import jsonlines
import numpy as np
import logging
import argparse

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

def document_transfer(data_collection):
    "Transfer the raw document into langchain supported format."
    documents = []
    for data, meta in data_collection:
        doc_id = str(uuid.uuid4())
        metadata = {"source": meta, "identify_id":doc_id}
        doc = Document(page_content=data, metadata=metadata)
        documents.append(doc)
    return documents

def document_append_id(documents):
    for _doc in documents:
        _doc.metadata["doc_id"] = _doc.metadata["identify_id"]
    return documents

def index_library(index_file_jsonl_path):
    index_list = []
    with open(index_file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            passages=[stu["context"][0],index_file_jsonl_path]
            index_list.append(passages)
    return index_list

def query_set(query_file_jsonl_path):
    query_list = []
    with open(query_file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            passages=stu["query"]
            query_list.append(passages)
    return query_list

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

    return metrics["MRR@1"], metrics["Hit@1"]

class Retrieval():
    def __init__(self,
                 vector_database="Chroma",
                 embedding_model="BAAI/bge-large-en-v1.5",
                 input_path = None,
                 retrieval_type = 'default',
                 append=True,
                 polish=False,
                 k=1,
                 fetch_k=1,
                 score_threshold=0.3,
                 reranker_model= "BAAI/bge-reranker-large",
                 top_n = 1,
                 enable_rerank = False,
                 **kwargs):

        self.vector_database = vector_database
        self.input_path = None
        self.retrieval_type = retrieval_type
        self.retriever = None
        self.k = k
        self.fetch_k = fetch_k
        self.score_threshold = score_threshold
        self.reranker_model= reranker_model,
        self.top_n = top_n
        self.enable_rerank=enable_rerank

        self.splitter = RecursiveCharacterTextSplitter(chunk_size= kwargs['child_size'] \
                    if 'child_size' in kwargs else 512)
        allowed_retrieval_type: ClassVar[Collection[str]] = (
            "default",
            "child_parent",
            'bm25',
        )

        if polish:
            self.polisher = QueryPolisher()
        else:
            self.polisher = None

        assert self.retrieval_type in allowed_retrieval_type, "search_type of {} not allowed.".format(   \
            self.retrieval_type)

        self.input_path = input_path
        assert self.input_path != None, "Should gave an input path!"

        try:
            if "instruct" in embedding_model:
                self.embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model)
            elif "bge" in embedding_model:
                self.embeddings = HuggingFaceBgeEmbeddings(
                    model_name=embedding_model,
                    encode_kwargs={'normalize_embeddings': True},
                    query_instruction="Represent this sentence for searching relevant passages:")
            elif "Google" == embedding_model:
                self.embeddings = GooglePalmEmbeddings()
            else:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    encode_kwargs={"normalize_embeddings": True},
                )
        except Exception as e:
            logging.error("Please select a proper embedding model.")
            logging.error(e)

        data_collection = index_library(self.input_path)
        logging.info("The parsing for the uploaded files is finished.")

        langchain_documents = document_transfer(data_collection)
        logging.info("The format of parsed documents is transferred.")

        if kwargs['search_type']=="similarity":
            kwargs['search_kwargs']={"k":self.k}
        elif kwargs['search_type']=="mmr":
            kwargs['search_kwargs']={"k":self.k, "fetch_k":self.fetch_k}
        elif kwargs['search_type']=="similarity_score_threshold":
            kwargs['search_kwargs']={"k":self.k, "score_threshold":self.score_threshold}

        if self.vector_database == "Chroma":
            self.database = Chroma
        elif self.vector_database == "Qdrant":
            self.database = Qdrant
        if self.retrieval_type == 'default':  # Using vector store retriever
            if append:
                knowledge_base = self.database.from_documents(documents=langchain_documents, embedding=self.embeddings,
                                                              **kwargs)
            else:
                knowledge_base = self.database.build(documents=langchain_documents, embedding=self.embeddings, **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                                              **kwargs)
            if self.vector_database == "Qdrant" and knowledge_base.is_local():
               # one local storage folder cannot be accessed by multiple instances of Qdrant client simultaneously.
               knowledge_base.client.close()
        elif self.retrieval_type == "child_parent":    # Using child-parent store retriever
            child_documents = self.splitter.split_documents(langchain_documents)
            langchain_documents = document_append_id(langchain_documents)
            if append:
                knowledge_base = self.database.from_documents(documents=langchain_documents, embedding=self.embeddings,
                                                              **kwargs)
                child_knowledge_base = self.database.from_documents(documents=child_documents, sign='child', \
                                                                    embedding=self.embeddings, **kwargs)
            else:
                knowledge_base = self.database.build(documents=langchain_documents, embedding=self.embeddings, **kwargs)
                child_knowledge_base = self.database.build(documents=langchain_documents, embedding=self.embeddings, \
                                            sign='child', **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                               child_document_store=child_knowledge_base, **kwargs)
            if self.vector_database == "Qdrant" :
                # one local storage folder cannot be accessed by multiple instances of Qdrant client simultaneously.
                if knowledge_base.is_local():
                    knowledge_base.client.close()
                if child_knowledge_base.is_local():
                    child_knowledge_base.client.close()
        elif self.retrieval_type == "bm25":
            self.docs = document_append_id(langchain_documents)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type,
                                              docs=self.docs,
                                              reranker_model=self.reranker_model,
                                              top_n = self.top_n,
                                              enable_rerank = self.enable_rerank,
                                              **kwargs)
        logging.info("The retriever is successfully built.")

    def pre_llm_inference_actions(self, model_name, query):
        if self.polisher:
            try:
                query = self.polisher.polish_query(model_name, query)
            except Exception as e:
                logging.info(f"Polish the user query failed, {e}")
                raise Exception("[Rereieval ERROR] query polish failed!")

        assert self.retriever is not None, logging.info("Please check the status of retriever")
        context = self.retriever.get_context(query)
        return context


class RetrieverAdapter():
    def __init__(self, retrieval_type='default', document_store=None, child_document_store=None, docs=None,  \
                 reranker_model="BAAI/bge-reranker-large", top_n = 1, enable_rerank = False, **kwargs):
        self.retrieval_type = retrieval_type
        if enable_rerank:
            from intel_extension_for_transformers.langchain_community.retrievers.bge_reranker import BgeReranker  # pylint: disable=E0401, E0611
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker(reranker_model)
            self.reranker = BgeReranker(model = reranker, top_n=top_n)
        else:
            self.reranker = None

        if self.retrieval_type == "default":
            self.retriever = VectorStoreRetriever(vectorstore=document_store, **kwargs)
        elif self.retrieval_type == "bm25":
            self.retriever = BM25Retriever.from_documents(docs, **kwargs)
        elif self.retrieval_type == "child_parent":
            self.retriever = ChildParentRetriever(parentstore=document_store, \
                                                  vectorstore=child_document_store,
                                                  **kwargs)  # pylint: disable=abstract-class-instantiated
        else:
            logging.error('The chosen retrieval type remains outside the supported scope.')

    def get_context(self, query):
        context = []
        retrieved_documents = self.retriever.get_relevant_documents(query)
        if self.reranker is not None:
            retrieved_documents = self.reranker.compress_documents(documents = retrieved_documents, query = query)
        for doc in retrieved_documents:
            context.append(doc.page_content)
        return context

def main():
    import os, shutil
    if os.path.exists("output"):
        shutil.rmtree("output", ignore_errors=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file_jsonl_path", type=str)
    parser.add_argument("--query_file_jsonl_path", type=str)
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

    args = parser.parse_args()

    index_file_jsonl_path = args.index_file_jsonl_path
    query_file_jsonl_path = args.query_file_jsonl_path
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

    query_list = query_set(query_file_jsonl_path)

    config = PipelineConfig(model_name_or_path=llm_model)
    build_chatbot(config)

    retrieval_results=[]
    for query in query_list:
        context=Retrieval(input_path=index_file_jsonl_path,
                         vector_database=vector_database,
                         embedding_model=embedding_model,
                         retrieval_type = retrieval_type,
                         polish = polish,
                         search_type=search_type,
                         k=k,
                         fetch_k=fetch_k,
                         score_threshold=score_threshold,
                         reranker_model=reranker_model,
                         top_n = top_n,
                         enable_rerank = enable_rerank
                         ).pre_llm_inference_actions(model_name=llm_model, query=query)
        retrieval_results.append(context)
    ground_truths=load_list(query_file_jsonl_path, "pos")
    MRR, Hit = evaluate(retrieval_results, ground_truths)

    file_json_path='result_retrieval.jsonl'

    if MRR and Hit:
        data = {
                "index_file_jsonl_path": args.index_file_jsonl_path,
                "query_file_jsonl_path": args.query_file_jsonl_path,
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
                "MRR": MRR,
                "Hit": Hit,
            }
        with jsonlines.open(file_json_path,"a") as file_json:
                file_json.write(data)

if __name__ == '__main__':
    main()
