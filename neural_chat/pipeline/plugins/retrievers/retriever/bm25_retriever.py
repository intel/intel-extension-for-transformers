from haystack.nodes import BM25Retriever
from neural_chat.plugins import register_plugin

class BM25Retriever():
    """Retrieve the document database with BM25 sparse algorithm."""
    
    def __int__(self, document_store = None, top_k = 1):
        assert document_store is not None, "Please give a document database for retrieving."
        self.retriever = BM25Retriever(document_store=document_store)

    def query_the_database(self, query):
        documents = retriever.retrieve(query)
        context = ""
        for doc in documents:
            context = context + doc.content + " "
        return context.strip()
