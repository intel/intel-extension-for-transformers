"""The wrapper for Chroma retriever with langchain"""

from langchain.vectorstores import Chroma

class ChromaRetriever():
    def __init__(self, persist_directory=None, embedding_function=None, search_type="mmr", search_kwargs=None):
        
        self.database = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        self.retriever = self.database.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
        
    
