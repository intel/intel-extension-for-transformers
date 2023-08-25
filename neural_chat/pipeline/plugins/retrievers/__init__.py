
from neural_chat.pipeline.plugins.intent_detector import IntentDetector
from neural_chat.pipeline.plugins.retrievers.indexing import DocumentIndexing
from neural_chat.pipeline.plugins.retrievers.retriever import Retriever
import transformers
import torch
from neural_chat.plugins import register_plugin
from neural_chat.pipeline.plugins.prompts.prompt import generate_qa_prompt, generate_prompt

@register_plugin("retriever")
class QA_Client():
    def __int__(self, persist_dir="./output", process=False, input_path=None, embedding_model="hkunlp/instructor-large", max_length=512, retrieval_type="dense", document_store=None, top_k=1, search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 5}):
        self.model = None
        self.tokenizer = None
        self.retrieval_type = retrieval_type
        
        self.intent_detector = IntentDetector()
        if os.path.exists(input_path):
            self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type, document_store=document_store,
                                               persist_dir=persist_dir, process=process,
                                               embedding_model=embedding_model, max_length=max_length)
            self.db = self.doc_parser.KB_construct(input_path)
            self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=self.db, top_k=top_k, search_type=search_type, search_kwargs=search_kwargs)
        
    
    
    def pre_llm_inference_actions(self, query, model, tokenizer, input_path):
        
        self.model = model
        self.tokenizer = tokenizer
        intent = self.intent_detector.predict_intent(query, model, tokenizer)
        
        if 'qa' not in intent.lower():
            print("Chat with AI Agent.")
            prompt = generate_prompt(query)
        else:
            print("Chat with QA agent.")
            context = self.retriever.get_context(query)
            prompt = generate_qa_prompt(query, context)
        return prompt
            
            