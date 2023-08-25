"""The wrapper for Chroma retriever based on langchain"""

from langchain.vectorstores import Chroma
from neural_chat.plugins import register_plugin

@register_plugin('chroma_retriever')
class ChromaRetriever():
    """Retrieve the document database with Chroma database using dense retrieval."""
    
    def __init__(self, database=None, search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}):
        assert document_store is not None, "Please give a document database for retrieving."
        self.retriever = database.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def query_the_database(self, query):
        documents = self.retriever.get_relevant_documents(query)
        context = ""
        for doc in documents: 
            context = context + doc.page_content + " "
        return context.strip()