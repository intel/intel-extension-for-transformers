"""Wrapper for parsing the uploaded user file and then make document indexing."""
import os
import re, json
from langchain.vectorstores.chroma import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from haystack.schema import Document as SDocument
from .utils import load_unstructured_data, laod_structured_data, get_chuck_data


class DocumentIndexing:
    def __init__(self, retrieval_type="dense", document_store=None, persist_dir="./output", process=False, embedding_model="hkunlp/instructor-large", max_length=512):
        """
        Wrapper for document indexing. Support dense and sparse indexing method.
        """
        self.retrieval_type = retrieval_type
        self.document_store = document_store
        self.process = process
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.max_length = max_length
        
        
    def parse_document(self, input):
        """
        Parse the uploaded file.
        """
        if input.endswith("pdf") or input.endswith("docx") or input.endswith("html") or input.endswith("txt") or input.endswith("md"):
            content = load_unstructured_data(input)
            if self.process:
                chuck = get_chuck_data(content, self.max_length, input)
            else:
                chuck = [[content.strip(),input]]
        elif input.endswith("jsonl") or input.endswith("xlsx"):
            chuck = laod_structured_data(input, self.process, self.max_length)
        else:
            print("This file is ignored. Will support this file format soon.")
        return chuck
        
    
    def batch_parse_document(self, input):
        """
        Parse the uploaded batch files in the input folder.
        """
        paragraphs = []
        for dirpath, dirnames, filenames in os.walk(input):
            for filename in filenames:
                if filename.endswith("pdf") or filename.endswith("docx") or filename.endswith("html") or filename.endswith(
                        "txt") or filename.endswith("md"):
                    content = load_unstructured_data(os.path.join(dirpath, filename))
                    if self.process:
                        chuck = get_chuck_data(content, self.max_length, input)
                    else:
                        chuck = [[content.strip(),input]]
                    paragraphs += chuck
                elif filename.endswith("jsonl") or filename.endswith("xlsx"):
                    chuck = laod_structured_data(os.path.join(dirpath, filename), self.process, self.max_length)
                    paragraphs += chuck
                else:
                    print("This file {} is ignored. Will support this file format soon.".format(filename))
        return paragraphs
    
    
    def KB_construct(self, input):
        """
        Construct the local knowledge base based on the uploaded file/files.
        """
        if self.retrieval_type == "dense":
            if os.path.exists(input):
                if os.path.isfile(input):
                    data_collection = self.parse_document(input)
                elif os.path.isdir(input):
                    data_collection = self.batch_parse_document(input)
                else:
                    print("Please check your upload file and try again!")
                    
                documents = []
                for data, meta in data_collection:
                    metadata = {"source": meta}
                    new_doc = Document(page_content=data, metadata=metadata)
                    documents.append(new_doc)
                embedding = HuggingFaceInstructEmbeddings(model_name=self.embedding_model)
                vectordb = Chroma.from_documents(documents=documents, embedding=embedding,
                                                 persist_directory=self.persist_dir)
                vectordb.persist()
                print("success")
                return vectordb
            else:
                print("There might be some errors, please wait and try again!")
        else:
            if os.path.exists(input):
                if os.path.isfile(input):
                    data_collection = self.parse_document(input)
                elif os.path.isdir(input):
                    data_collection = self.batch_parse_document(input)
                else:
                    print("Please check your upload file and try again!")
                if self.document_store == "inmemory":
                    document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
                elif self.document_stor == "Elasticsearch":
                    document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_index_1",
                                                                port=9200, search_fields=["content", "title"])

                documents = []
                for data, meta in data_collection:
                    metadata = {"source": meta}
                    new_doc = SDocument(content=data, metadata=metadata)
                    documents.append(new_doc)
                document_store.write_documents(documents)
                print("success")
                return document_store
            else:
                print("There might be some errors, please wait and try again!")