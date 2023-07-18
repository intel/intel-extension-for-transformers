from haystack.document_stores import InMemoryDocumentStore,ElasticsearchDocumentStore

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi import APIRouter
import uvicorn
import argparse
import collections
import json
import time
import uuid
import shutil
import base64
import os
from typing import Any, Dict
from haystack.telemetry import send_event_if_public_demo
import doc_index
from embedding_xlsx import load_xlsx_file, split_paragraph, sp_split_paragraph
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, pipeline
from logger import build_logger
from haystack.nodes import MarkdownConverter, TextConverter
from utils import detect_language

def ask_gm_documents_dense_embedding(folder_path, process_content=False):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # print(os.path.join(dirpath, filename))
            if filename.endswith(".jsonl"):
                documents = doc_index.d_load_jsonl_file(os.path.join(dirpath, filename), process_content)
            elif filename.endswith(".pdf") or filename.endswith(".docx") or filename.endswith(".txt"):
                documents = doc_index.d_load_file(os.path.join(dirpath, filename), process_content)
            elif filename.endswith(".xlsx"):
                documents = doc_index.d_load_xlsx(os.path.join(dirpath, filename), process_content)
            else:
                print("{} is ignored. Will support this file format soon.".format(filename))
                continue
    doc_index.persist_embedding(documents, "/tmp/ask_gm_dense_retrieval_chinese",
                                model_path="shibing624/text2vec-base-chinese")
    doc_index.persist_embedding(documents, "/tmp/ask_gm_dense_retrieval_english",
                                model_path="hkunlp/instructor-large")

def ask_gm_documents_sparse_embedding(folder_path, process_content=False):
    document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_askgm_sparse",
                        port=9200, search_fields=["content", "title"])

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith("jsonl"):
                document_store = doc_index.s_load_jsonl_file(
                    os.path.join(dirpath, filename), process_content, document_store)
            elif filename.endswith("pdf") or filename.endswith("docx") or filename.endswith("txt"):
                document_store = doc_index.s_load_file(
                    os.path.join(dirpath, filename), process_content, document_store)
            elif filename.endswith(".xlsx"):
                document_store = doc_index.s_load_xlsx(
                    os.path.join(dirpath, filename), process_content, document_store)
            else:
                print("{} is ignored. Will support this file format soon.".format(filename))
                continue

from elasticsearch import Elasticsearch
es = Elasticsearch(hosts=["localhost:9200"])
index_name = "elastic_askgm_sparse"
index_exists = es.indices.exists(index=index_name)
if not index_exists:
    ask_gm_documents_sparse_embedding("./doc/")

if not os.path.exists("/tmp/ask_gm_dense_retrieval_chinese") and \
   not os.path.exists("/tmp/ask_gm_dense_retrieval_english"):
    ask_gm_documents_dense_embedding("./doc/")

inc_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
es_document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_index_1",
                                               port=9200, search_fields=["content", "title"])

logger = build_logger("fastrag_service", f"fastrag_service.log")

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=80)
parser.add_argument("--model-path", type=str, default="mosaicml/mpt-7b-chat")
args = parser.parse_args()
logger.info(f"args: {args}")

from haystack.schema import Document

# 3 example documents to index
from inc_document import inc_examples
inc_documents = []
for i, d in enumerate(inc_examples):
    inc_documents.append(Document(content=d["doc"], id=i))

inc_document_store.write_documents(inc_documents)


from haystack.nodes import BM25Retriever, SentenceTransformersRanker
from llm_invocation import FastRAGPromptModel
from chatglm_invocation import ChatGLMPromptModel
from mpt_invocation import MptPromptModel
inc_retriever = BM25Retriever(document_store=inc_document_store, top_k = 10)
es_retriever = BM25Retriever(document_store=es_document_store, top_k = 10)

reranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32, top_k=5)

# FastRAG model wrapper
#pmodel = FastRAGPromptModel(model_name_or_path=args.model_path)
#pmodel = ChatGLMPromptModel(model_name_or_path=args.model_path)
pmodel = MptPromptModel(model_name_or_path=args.model_path)

# langchain model
langchain_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
langchain_model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 low_cpu_mem_usage=True,
                                                 return_dict=True,
                                                 max_seq_len=8192,
                                                 trust_remote_code=True)
#langchain_model = AutoModel.from_pretrained(args.model_path)
langchain_pipe = pipeline("text-generation", model=langchain_model,
                          tokenizer=langchain_tokenizer, max_new_tokens=256)

from haystack.nodes.prompt.prompt_node import PromptNode
prompt = PromptNode(model_name_or_path = pmodel)

from haystack.nodes.other.shaper import Shaper
shaper = Shaper(func="join_documents", inputs={"documents": "documents"}, outputs=["documents"])

from haystack import Pipeline

inc_pipeline = Pipeline()
inc_pipeline.add_node(component=inc_retriever, name="Retriever", inputs=["Query"])
inc_pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
inc_pipeline.add_node(component=shaper, name="Shaper", inputs=["Reranker"])

wiki_pipeline = Pipeline()
wiki_pipeline.add_node(component=es_retriever, name="Retriever", inputs=["Query"])
wiki_pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
wiki_pipeline.add_node(component=shaper, name="Shaper", inputs=["Reranker"])

from haystack.nodes.prompt.prompt_node import PromptTemplate
custom_template = PromptTemplate(
    name="llama-instruct",
    prompt_text="Given the context please answer the question. Context: $documents; Question: $query; Answer:")
prompt.add_prompt_template(custom_template)
prompt.default_prompt_template = custom_template
inc_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])
wiki_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])


#from chatglm import ChatLLM
#chatLLM = ChatLLM()
#chatLLM.history = []
#chatLLM.load_model(model_name_or_path="./chatglm-6b")
#chatLLM.temperature = 0.05
#chatLLM.top_p = 0.9



app = FastAPI()
router = APIRouter()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseConfig, BaseModel, Extra, Field
from typing import Dict, List, Optional, Union
class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class QueryRequest(RequestBaseModel):
    query: str
    domain: str
    blob: Optional[str]
    filename: Optional[str]
    embedding: Optional[str] = 'dense'
    params: Optional[dict] = None
    debug: Optional[bool] = False


class QueryResponse(BaseModel):
    query: str
    answers: Optional[List] = []
    documents: List[Document] = []
    images: Optional[Dict] = None
    relations: Optional[List] = None
    debug: Optional[Dict] = Field(None, alias="_debug")
    timings: Optional[Dict] = None
    results: Optional[List] = None

@router.post("/fastrag/query")
def query(request: QueryRequest):
    domain = request.domain
    logger.info(f"fastrag query received for {domain}")
    if domain == "INC":
        result = _process_request(inc_pipeline, request)
    elif domain == "WIKI":
        result = _process_request(wiki_pipeline, request)
    elif domain == "ASK_GM":
        if request.embedding == "dense":
            if detect_language(request.query) == 'English':
                output_path = '/tmp/ask_gm_dense_retrieval_english'
                embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
            else:
                output_path = '/tmp/ask_gm_dense_retrieval_chinese'
                embeddings = HuggingFaceInstructEmbeddings(model_name="shibing624/text2vec-base-chinese")
            vectordb = Chroma(persist_directory=output_path, embedding_function=embeddings)
            retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs = {"k": 1, "fetch_k": 5})
            prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:"""
            from langchain.prompts import PromptTemplate
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            docs = retriever.get_relevant_documents(request.query)
            logger.info("dense retreival done...")
            hf = HuggingFacePipeline(pipeline=langchain_pipe)
            chain = load_qa_chain(hf, chain_type="stuff", prompt=PROMPT)
            chain_result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
            print(chain_result)
            result = _process_dense_retrieval(chain_result)
        elif request.embedding == 'sparse':
            document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_askgm_sparse", port=9200, search_fields=["content", "title"])
            askgm_pipeline = Pipeline()
            askgm_retriever = BM25Retriever(document_store=document_store, top_k = 10)
            askgm_pipeline.add_node(component=askgm_retriever, name="Retriever", inputs=["Query"])
            askgm_pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
            askgm_pipeline.add_node(component=shaper, name="Shaper", inputs=["Reranker"])
            from haystack.nodes.prompt.prompt_node import PromptNode
            from haystack.nodes.prompt.prompt_node import PromptTemplate
            prompt = PromptNode(model_name_or_path = pmodel)
            custom_template = PromptTemplate(
                         name="llama-instruct",
                         prompt_text="Given the context please answer the question. Context: $documents; Question: $query; Answer:")
            prompt.add_prompt_template(custom_template)
            prompt.default_prompt_template = custom_template
            askgm_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])
            logger.info("sparse retrieval done...")
            result = _process_request(askgm_pipeline, request)
    elif domain == "Customized":
        if request.blob:
            file_content = base64.b64decode(request.blob)
            random_suffix = str(uuid.uuid4().hex)
            file_path = f"/tmp/customized_doc_{random_suffix}" + request.filename
            with open(file_path, "wb") as f:
                f.write(file_content)

        if request.filename.endswith("md"):
            converter = MarkdownConverter()
            customized_documents = converter.convert(file_path=file_path)
            customized_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
            customized_document_store.write_documents(customized_documents)
        elif request.filename.endswith("txt"):
            converter = TextConverter()
            customized_documents = converter.convert(file_path=file_path)
            customized_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
            customized_document_store.write_documents(customized_documents)
        else:
            if request.embedding == "dense":  # currently use Chroma as the dense retrieval datastore
                if request.filename.endswith("jsonl"):
                    documents = doc_index.d_load_jsonl_file(file_path, process=True, max_length=4096)
                elif request.filename.endswith("pdf") or request.filename.endswith("docx"):
                    documents = doc_index.d_load_file(file_path, process=True, max_length=4096)
                random_suffix = str(uuid.uuid4().hex)
                output_path = f"/tmp/dense_retrieval_{random_suffix}"
                doc_index.persist_embedding(documents, output_path, model_path="hkunlp/instructor-large")
                embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
                vectordb = Chroma(persist_directory=output_path, embedding_function=embeddings)
                retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs = {"k": 1, "fetch_k": 5})
                prompt_template = """Below is an instruction that describes a task,
                                    paired with an input that provides further context.
                                    Write a response that appropriately completes the request.\n\n
                                    ### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:"""
                from langchain.prompts import PromptTemplate
                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )
                docs = retriever.get_relevant_documents(request.query)
                hf = HuggingFacePipeline(pipeline=langchain_pipe)
                chain = load_qa_chain(hf, chain_type="stuff", prompt=PROMPT)
                chain_result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
                print(chain_result)
                result = _process_dense_retrieval(chain_result)
            elif request.embedding == "sparse":   # sparse retrieval datastores has inmemory and Elasticsearch
                customized_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
                if request.filename.endswith("jsonl"):
                    customized_document_store = doc_index.s_load_jsonl_file(file_path, args.process,
                                                                            customized_document_store)
                elif request.filename.endswith("pdf") or request.filename.endswith("docx"):
                    customized_document_store = doc_index.s_load_file(file_path, args.process,
                                                                    customized_document_store)
        customized_pipeline = Pipeline()
        customized_retriever = BM25Retriever(document_store=customized_document_store, top_k = 10)
        customized_pipeline.add_node(component=customized_retriever, name="Retriever", inputs=["Query"])
        customized_pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
        customized_pipeline.add_node(component=shaper, name="Shaper", inputs=["Reranker"])
        customized_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])
        result = _process_request(customized_pipeline, request)
    return result

app.include_router(router)

@send_event_if_public_demo
def _process_dense_retrieval(result):
    text = result['output_text'].replace('\n', '')
    #text = result['result'].replace('\n', '')
    print("=================", text)
    def stream_results():
        yield "data: {}\n\n".format(text)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")

@send_event_if_public_demo
def _process_request(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()

    params = request.params or {}

    # format global, top-level filters (e.g. "params": {"filters": {"name": ["some"]}})
    if "filters" in params.keys():
        params["filters"] = _format_filters(params["filters"])

    # format targeted node filters (e.g. "params": {"Retriever": {"filters": {"value"}}})
    for key in params.keys():
        if isinstance(params[key], collections.Mapping) and "filters" in params[key].keys():
            params[key]["filters"] = _format_filters(params[key]["filters"])

    prediction  = pipeline.run(query=request.query, params=params, debug=request.debug)
    # Ensure answers and documents exist, even if they're empty lists
    if not "documents" in prediction:
        prediction["documents"] = []
    if not "answers" in prediction:
        prediction["answers"] = []

    def stream_results():
        for idx, result in enumerate(prediction["results"]):
            #data = json.dumps({"text": result, "error_code": 0}).encode() + b"\0"
            yield f"data: {result}\n\n"
        yield f"data: [DONE]\n\n"

    logger.info(
        json.dumps(
            {
                "request": request,
                "response": prediction,
                "time": f"{(time.time() - start_time):.2f}",
            },
            default=str,
        )
    )

    return StreamingResponse(stream_results(), media_type="text/event-stream")

def _format_filters(filters):
    """
    Adjust filters to compliant format:
    Put filter values into a list and remove filters with null value.
    """
    new_filters = {}
    if filters is None:
        logger.warning(
            f"Request with deprecated filter format ('\"filters\": null'). "
            f"Remove empty filters from params to be compliant with future versions"
        )
    else:
        for key, values in filters.items():
            if values is None:
                logger.warning(
                    f"Request with deprecated filter format ('{key}: null'). "
                    f"Remove null values from filters to be compliant with future versions"
                )
                continue

            if not isinstance(values, list):
                logger.warning(
                    f"Request with deprecated filter format ('{key}': {values}). "
                    f"Change to '{key}':[{values}]' to be compliant with future versions"
                )
                values = [values]

            new_filters[key] = values
    return new_filters


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

