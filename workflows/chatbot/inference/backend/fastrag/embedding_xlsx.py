import os
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain.docstore.document import Document
import re, json
import pandas as pd
import argparse
from haystack.schema import Document as SDocument
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore


## CUDA_VISIBLE_DEVICES=5 python embedding_xlsx.py
def load_xlsx_file(file_path):
    df = pd.read_excel(file_path)

    df1 = df.loc[(df["Answers _Chinese"].notnull()) | (df["Answers_ English"].notnull())]
    # list(df.columns)
    all_data = []
    chinese = []
    english = []
    for index, row in df1.iterrows():
        sub0 = "User Query: " + row['Questions _ English'] + "Answer: " + row["Answers_ English"]
        sub1 = "用户问询： " + row['Questions _ Chinese'] + "回复:" + row["Answers _Chinese"]
        chinese.append(sub1)
        english.append(sub0)
    all_data = chinese + english
    return all_data


def split_paragraph(text):
    documents = []
    for c_index, data in enumerate(text):
        data.replace('#', " ")
        data = re.sub(r'\s+', ' ', data)
        new_doc = Document(page_content=data, metadata={"source": c_index})
        documents.append(new_doc)
    return documents


def sp_split_paragraph(text, document_store):
    documents = []
    for c_index, data in enumerate(text):
        data.replace('#', " ")
        data = re.sub(r'\s+', ' ', data)
        new_doc = SDocument(content=data, meta={"source": c_index})
        documents.append(new_doc)
    document_store.write_documents(documents)
    return document_store



def persist_embedding(documents, persist_directory, model_path):
    ## persistly save the local file into disc
    embedding = HuggingFaceInstructEmbeddings(model_name=model_path)
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None

if __name__ == "__main__":
    file_path = "/data1/lkk/llm_inference/chat-langchain/test/test.xlsx"

    # content = load_pdf(pdf_path)
    content = load_xlsx_file(file_path)
    documents = split_paragraph(content)
    persist_embedding(documents)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='The user upload file.',
                        default="/data1/lkk/llm_inference/chat-langchain/test/test.xlsx")
    parser.add_argument('--embedding_model', type=str, help='Select which model to embed the content.', default='/data1/lkk/instructor_large/')
    parser.add_argument('--output_path', type=str, help='Where to save the embedding.', default='db_xlsx_new15')
    parser.add_argument('--embedding_method', type=str, help='Select to use dense retrieval or sparse retrieval.', default='dense')
    parser.add_argument('--store', type=str, help='Select to use dense retrieval or sparse retrieval.',
                        default='dense')

    args = parser.parse_args()

    content = load_xlsx_file(file_path)
    documents = split_paragraph(content)
    persist_embedding(documents)

    if args.embedding_method == "dense":  # currently use Chroma as the dense retrieval datastore
        content = load_xlsx_file(file_path)
        documents = split_paragraph(content)
        persist_embedding(documents, args.output_path, args.embedding_model)
    elif args.embedding_method == "sparse":   # sparse retrieval datastores has inmemory and Elasticsearch
        if args.store == "inmemory":
            document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
        elif args.store == "Elasticsearch":
            document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_index_1",
                                               port=9200, search_fields=["content", "title"])
        # import pdb;pdb.set_trace()
        # if args.file_path.endswith("jsonl"):
        #     document_store = s_load_jsonl_file(args.file_path, args.process, document_store)
        # elif args.file_path.endswith("pdf") or args.file_path.endswith("docx"):
        #     document_store = s_load_file(args.file_path, args.process, document_store)
        content = load_xlsx_file(file_path)
        document_store = sp_split_paragraph(content, document_store)
        if args.store == "Elasticsearch": # only Elasticsearch could be saved locally, inmemory should be load in the memory
            document_store.save(index_path="my_index.faiss")
        else:
            print("in memory db is done")
