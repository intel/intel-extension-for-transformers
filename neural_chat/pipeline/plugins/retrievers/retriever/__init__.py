"""The class defination for the retriever. Supporting langchain-based and haystack-based retriever."""

# from langchain.retrievers.elastic_search_bm25 import ElasticSearchBM25Retriever
# from .langchain import ChromaRetriever
#
# from fastrag.retrievers import BaseRetriever, ColBERTRetriever
# from haystack.nodes import BM25Retriever

from .bm25_retriever import BM25Retriever
from .chroma_retriever import ChromaRetriever



class Retriever():
    """Retrieve the document database with BM25 sparse algorithm."""

    def __int__(self, retrieval_type="dense", document_store=None, top_k=1, search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}):
        self.retrieval_type = retrieval_type
        if not document_store:
            raise ValueError(f"Please give a knowledge base for retrieval.")
        if retrieval_type == "dense":
            self.retriever = ChromaRetriever(database=document_store, search_type=search_type, search_kwargs=search_kwargs).retriever
        else:
            self.retriever = BM25Retriever(document_store=document_store, top_k=top_k).retriever

    def get_context(self, query):
        context = self.retriever.query_the_database(query)
        return context