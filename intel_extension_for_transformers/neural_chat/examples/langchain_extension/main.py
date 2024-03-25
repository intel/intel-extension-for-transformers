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

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from intel_extension_for_transformers.langchain_community.vectorstores import Chroma
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.parser.parser import DocumentParser
import requests

url = "https://d1io3yog0oux5.cloudfront.net/_897efe2d574a132883f198f2b119aa39/intel/db/888/8941/file/412439%281%29_12_Intel_AR_WR.pdf"
filename = "Intel_AR_WR.pdf"
response = requests.get(url)
with open(filename, 'wb') as file:
    file.write(response.content)
print(f"File '{filename}' downloaded successfully.")

document_parser = DocumentParser()
input_path="./Intel_AR_WR.pdf"
data_collection=document_parser.load(input=input_path)
documents = []
for data, meta in data_collection:
    doc = Document(page_content=data, metadata={"source":meta})
    documents.append(doc)
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
knowledge_base = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory='./output')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
pipe = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128))
retriever = VectorStoreRetriever(vectorstore=knowledge_base)
retrievalQA = RetrievalQA.from_llm(llm=pipe, retriever=retriever)
result = retrievalQA({"query": "What is IDM 2.0?"})
print(result)
