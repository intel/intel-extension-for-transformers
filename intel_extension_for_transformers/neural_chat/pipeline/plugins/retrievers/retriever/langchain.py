"""The wrapper for Chroma retriever based on langchain"""

from langchain.vectorstores import Chroma

class ChromaRetriever():
    def __init__(self, database, search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}):
        self.retriever = database.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
