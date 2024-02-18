# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import re
import tempfile
from typing import Any, Dict
from threading import Thread
from haystack.telemetry import send_event_if_public_demo
import doc_index
from embedding_xlsx import load_xlsx_file, split_paragraph, sp_split_paragraph
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, GenerationConfig, TextIteratorStreamer
from logger import build_logger
from haystack.nodes import MarkdownConverter, TextConverter
from utils import detect_language
from database.mysqldb import MysqlDb
from starlette.responses import RedirectResponse
from mysqldb import MysqlDb

logger = build_logger("fastrag_service", f"fastrag_service.log")
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=80)
parser.add_argument("--model-path", type=str, default="mosaicml/mpt-7b-chat")
parser.add_argument(
    "--cache-chat-config-file", default="./llmcache/cache_config.yml", help="the cache config file"
)
parser.add_argument(
    "--cache-embedding-model-dir", default="GanymedeNil/text2vec-large-chinese", help="the cache embedding model directory"
)
args = parser.parse_args()
logger.info(f"args: {args}")

def ask_gm_documents_dense_embedding(folder_path, process_content=False):
    documents = []
    with tempfile.TemporaryDirectory(dir="/tmp/my_subdirectory") as temp_dir:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(".json"):
                    documents = doc_index.d_load_jsonl_file(os.path.join(dirpath, filename), process_content, documents)
                elif filename.endswith(".xlsx"):
                    documents = doc_index.d_load_xlsx(os.path.join(dirpath, filename), process_content)
                else:
                    print("{} is ignored. Will support this file format soon.".format(filename))
                    continue
        doc_index.persist_embedding(documents, temp_dir, model_path="shibing624/text2vec-large-chinese")
        doc_index.persist_embedding(documents, temp_dir, model_path="hkunlp/instructor-large")

def ask_gm_documents_sparse_embedding(folder_path, process_content=False):
    document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_askgm_sparse",
                        port=9200, search_fields=["content", "title"])
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith("json"):
                document_store = doc_index.s_load_jsonl_file(
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
index_exists = es.indices.exists(index=index_name)
if not index_exists:
    ask_gm_documents_sparse_embedding("./doc/ask_gm/", True)

if not os.path.exists("/tmp/ask_gm_dense_retrieval_chinese") and \
   not os.path.exists("/tmp/ask_gm_dense_retrieval_english"):
    ask_gm_documents_dense_embedding("./doc/ask_gm/", True)

inc_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
es_document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_index_1",
                                               port=9200, search_fields=["content", "title"])

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

reranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32, top_k=1)

# FastRAG model wrapper
#pmodel = FastRAGPromptModel(model_name_or_path=args.model_path)
#pmodel = ChatGLMPromptModel(model_name_or_path=args.model_path)
pmodel = MptPromptModel(model_name_or_path=args.model_path)


# langchain model
from mpt_wrapper import MosaicML
import torch
import intel_extension_for_pytorch as ipex
config = AutoConfig.from_pretrained("./mpt-7b-chat", trust_remote_code=True)
config.attn_config['attn_impl'] = "torch"
config.init_device = 'cuda:0' if torch.cuda.is_available() else "cpu"
langchain_model = AutoModelForCausalLM.from_pretrained(
    "./mpt-7b-chat",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    config=config,
)
langchain_tok = AutoTokenizer.from_pretrained("./mpt-7b-chat", trust_remote_code=True)
langchain_model = ipex.optimize(langchain_model, dtype=torch.bfloat16)
stop_token_ids = langchain_tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
stop_token_ids.append(langchain_model.generation_config.eos_token_id)
stop_token_ids.append(langchain_tok(".", return_tensors="pt").input_ids)
stop_token_ids.append(langchain_tok("!", return_tensors="pt").input_ids)
stop_token_ids.append(langchain_tok("。", return_tensors="pt").input_ids)
stop_token_ids.append(langchain_tok("！", return_tensors="pt").input_ids)
langchain_tok.pad_token = langchain_tok.eos_token
langchain_tok.add_special_tokens({'pad_token': '[PAD]'})
with tempfile.TemporaryDirectory(dir="/tmp/my_subdirectory") as temp_dir:
    if not os.path.exists(temp_dir):
        documents = doc_index.d_load_young_pat_xlsx("./doc/young_pat/pat.xlsx", True)
        doc_index.persist_embedding(documents, temp_dir, model_path="hkunlp/instructor-large")

with tempfile.TemporaryDirectory(dir="/tmp/my_subdirectory") as temp_dir:
    english_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    chinese_embeddings = HuggingFaceInstructEmbeddings(model_name="shibing624/text2vec-base-chinese")

    young_pat_vectordb = Chroma(persist_directory=temp_dir,
                                embedding_function=english_embeddings)
    young_pat_dense_retriever = young_pat_vectordb.as_retriever(search_type="mmr",
                                                                search_kwargs={"k": 2, "fetch_k": 5})

    ask_gm_eng_vectordb = Chroma(persist_directory=temp_dir,
                                 embedding_function=english_embeddings)
    ask_gm_eng_retriever = ask_gm_eng_vectordb.as_retriever(search_type="mmr",
                                                            search_kwargs={"k": 2, "fetch_k": 5})

    ask_gm_chn_vectordb = Chroma(persist_directory=temp_dir,
                                 embedding_function=chinese_embeddings)
    ask_gm_chn_retriever = ask_gm_chn_vectordb.as_retriever(search_type="mmr",
                                                            search_kwargs={"k": 2, "fetch_k": 5})


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class StopOnTokensWithPeriod(StoppingCriteria):
    def __init__(self, min_length: int, start_length: int, stop_token_id: list[int]):
        self.min_length = min_length
        self.start_length = start_length
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if scores is not None:
            if len(scores) > self.min_length:
                for stop_id in self.stop_token_id:
                    if input_ids[0][self.start_length - 1 + len(scores)] == stop_id:
                        return True
        elif input_ids.shape[-1] - self.start_length > self.min_length:
            for stop_id in self.stop_token_id:
                if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                    return True
        return False

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
    name="llama-instruct", prompt_text="""Have a conversation with a human, answer the following questions as best you can. You can refer to the following document and context.\n### Question: $query\n### Context: $documents\n### Response:""")
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


class FeedbackRequest(RequestBaseModel):
    """
    Request class for feedback api
    'feedback_id' is set to be auto_increment, no need to pass as argument
    """
    # feedback_id: Optional[int] = None
    question: str
    answer: str
    feedback: Optional[int] = 0


class QueryResponse(BaseModel):
    query: str
    answers: Optional[List] = []
    documents: List[Document] = []
    images: Optional[Dict] = None
    relations: Optional[List] = None
    debug: Optional[Dict] = Field(None, alias="_debug")
    timings: Optional[Dict] = None
    results: Optional[List] = None

def dense_retrieval_execution(request, docs):
    documents=[]
    for doc in docs:
        print(doc.page_content)
        documents.append(doc.page_content)
    context = " ".join(documents)
    prompt= """Have a conversation with a human, answer the following questions as best you can.
            You can refer to the following document and context.\n Question:\n{}\n Context:\n{}\n Response:""".format(request.query, context)
    stop = StopOnTokens()
    inputs = langchain_tok(prompt, return_tensors="pt", padding=True)
    inputs = inputs.to(langchain_model.device)
    generate_kwargs = dict(
        use_cache=True,
        max_new_tokens=378,
        temperature=0.001,
        top_p=0.9,
        top_k=1,
        repetition_penalty=1.0,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    generation_config = GenerationConfig(**generate_kwargs)
    start_time = time.time()
    with torch.no_grad():
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            output = langchain_model.generate(**inputs, generation_config=generation_config)
        output = output[:, inputs.input_ids.shape[1]:]
    generated_texts = langchain_tok.batch_decode(output, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)[0]
    stopwords=["<|im_end|>", "<|endoftext|>"]
    def enforce_stop_tokens(text: str) -> str:
        """Cut off the text as soon as any stop words occur."""
        return re.split("|".join(stopwords), text)[0]
    generated_texts = enforce_stop_tokens(generated_texts)
    end_time = time.time()
    print("inference cost {} seconds.".format(end_time - start_time))
    result = _process_dense_retrieval(request, generated_texts)
    return result

def convert_query_to_text(query, documents):

    context = " ".join(documents)
    prompt= """Have a conversation with a human, answer the following questions as best you can.
        You can refer to the following document and context.\n Question:\n{}\n Context:\n{}\n Response:""".format(query, context)

    return prompt


def convert_query_chat_text(query):
    prompt = """Have a conversation with a human. You are required to generate suitable response in short to the user input.
    \n### Input:\n{}\n### Response:""".format(query)

    return prompt

def intent_detect_optimize(query):
    # prompt = "Please help me to refine the below prompt to make it easy to understand.\n Prompt:{}".format(query)
    prompt= """Please identify the intent of the provided context. You must only respond with "chitchat" or "QA" without explanations or engaging in conversation.\nContext:{}\nIntent:""".format(query)
    input_ids = langchain_tok(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(langchain_model.device)
    streamer = TextIteratorStreamer(langchain_tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    stop = StopOnTokens()
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=5,
        temperature=0.001,
        do_sample=True,
        top_k=1,
        repetition_penalty=1.0,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([stop]),
    )
    def generate_output():
        with torch.no_grad():
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                return langchain_model.generate(**generate_kwargs)
    thread = Thread(target=generate_output)
    thread.start()
    text = ""
    for new_text in streamer:
        text += new_text
    return text

def sparse_chitchat(request):
    prompt_input = convert_query_chat_text(request.query)
    input_tokens = langchain_tok.batch_encode_plus(
        [prompt_input], return_tensors="pt", padding=True
    )
    input_token_len = input_tokens.input_ids.shape[-1]
    stop = StopOnTokensWithPeriod(min_length=108, start_length=input_token_len, stop_token_id=stop_token_ids)
    generate_kwargs = dict(
        eos_token_id=0,
        pad_token_id=0,
        use_cache=True,
        max_new_tokens=128,
        temperature=0.3,
        top_p=0.9,
        top_k=1,
        repetition_penalty=1.1,
        num_beams=1,
        return_dict_in_generate=True,
        stopping_criteria=StoppingCriteriaList([stop]))
    with torch.no_grad():
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            generation_output = langchain_model.generate(**input_tokens, **generate_kwargs)
    generated_texts = langchain_tok.decode(generation_output.sequences[0], skip_special_tokens=True)
    if "### Response:" in generated_texts:
        generated_texts = generated_texts.split("### Response:")[1].strip()
    print("generated_texts=========", generated_texts)

    def stream_results():
        yield "data: {}\n\n".format(generated_texts)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")

#@router.post("/fastrag/query")
@router.post("/v1/chat/completions")
def query(request: QueryRequest):
    domain = request.domain
    logger.info(f"fastrag query received for {domain}")
    result = None
    if domain == "INC":
        result = _process_request(inc_pipeline, request)
    elif domain == "WIKI":
        result = _process_request(wiki_pipeline, request)
    elif domain == "ASK_GM":
        if request.embedding == "dense":
            if detect_language(request.query) == 'Chinese':
                docs = ask_gm_chn_retriever.get_relevant_documents(request.query)
            else:
                docs = ask_gm_eng_retriever.get_relevant_documents(request.query)
            print("docs========", docs)
            result = dense_retrieval_execution(request, docs)
        elif request.embedding == 'sparse':
            time_1 = time.time()
            intent = intent_detect_optimize(request.query)
            time_2 = time.time()
            print("intent========", intent)
            if 'qa' not in intent.lower():
                intent = "chitchat"
            print("Intent detection time is {} seconds and the intent is {}".format(time_2-time_1, intent))
            if "chitchat" in intent.lower():
                return sparse_chitchat(request)
            else:
                askgm_document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_askgm_sparse", port=9200, search_fields=["content", "title"])
                askgm_pipeline = Pipeline()
                askgm_retriever = BM25Retriever(document_store=askgm_document_store, top_k = 10)
                askgm_pipeline.add_node(component=askgm_retriever, name="Retriever", inputs=["Query"])
                askgm_pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
                askgm_pipeline.add_node(component=shaper, name="Shaper", inputs=["Reranker"])
                askgm_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])
                result = _process_request(askgm_pipeline, request)
    elif domain == "Young_Pat":
        if request.embedding == "dense":
            docs = young_pat_dense_retriever.get_relevant_documents(request.query)
            print("docs========", docs)
            result = dense_retrieval_execution(request, docs)
        else:
            time_1 = time.time()
            intent = intent_detect_optimize(request.query)
            time_2 = time.time()
            print("intent========", intent)
            if 'qa' not in intent.lower():
                intent = "chitchat"
            print("Intent detection time is {} seconds and the intent is {}".format(time_2-time_1, intent))
            if "chitchat" in intent.lower():
                return sparse_chitchat(request)
            else:
                customized_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
                customized_document_store = doc_index.s_load_young_pat_xlsx("./doc/young_pat/pat.xlsx", True,
                                                                            customized_document_store)
                young_pat_pipeline = Pipeline()
                young_pat_retriever = BM25Retriever(document_store=customized_document_store, top_k = 10)
                young_pat_pipeline.add_node(component=young_pat_retriever, name="Retriever", inputs=["Query"])
                young_pat_pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
                young_pat_pipeline.add_node(component=shaper, name="Shaper", inputs=["Reranker"])
                young_pat_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])
                result = _process_request(young_pat_pipeline, request)
    elif domain == "Customized":
        if request.blob:
            file_content = base64.b64decode(request.blob)
            random_suffix = str(uuid.uuid4().hex)
            sanitized_filename = os.path.basename(request.filename)
            file_path = f"/tmp/customized_doc_{random_suffix}_{sanitized_filename}"
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
                documents = []
                if request.filename.endswith("jsonl") or request.filename.endswith("json"):
                    documents = doc_index.d_load_jsonl_file(file_path, True, documents)
                elif request.filename.endswith("pdf") or request.filename.endswith("docx"):
                    documents = doc_index.d_load_file(file_path, True)
                elif request.filename.endswith(".xlsx") or request.filename.endswith(".csv"):
                    documents = doc_index.d_load_xlsx(file_path, True)
                random_suffix = str(uuid.uuid4().hex)
                output_path = f"/tmp/dense_retrieval_{random_suffix}"
                doc_index.persist_embedding(documents, output_path, model_path="hkunlp/instructor-large")
                embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
                vectordb = Chroma(persist_directory=output_path, embedding_function=embeddings)
                retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs = {"k": 2, "fetch_k": 5})
                docs = retriever.get_relevant_documents(request.query)
                result = dense_retrieval_execution(request, docs)
                return result
            elif request.embedding == "sparse":   # sparse retrieval datastores has inmemory and Elasticsearch
                customized_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
                if request.filename.endswith("jsonl") or request.filename.endswith("json"):
                    customized_document_store = doc_index.s_load_jsonl_file(file_path, True,
                                                                            customized_document_store)
                elif request.filename.endswith("pdf") or request.filename.endswith("docx"):
                    customized_document_store = doc_index.s_load_file(file_path, True,
                                                                    customized_document_store)
                elif request.filename.endswith(".xlsx") or request.filename.endswith(".csv"):
                    customized_document_store = doc_index.s_load_xlsx(file_path, True,
                                                                      customized_document_store)
        customized_pipeline = Pipeline()
        customized_retriever = BM25Retriever(document_store=customized_document_store, top_k = 10)
        customized_pipeline.add_node(component=customized_retriever, name="Retriever", inputs=["Query"])
        customized_pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
        customized_pipeline.add_node(component=shaper, name="Shaper", inputs=["Reranker"])
        customized_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])
        result = _process_request(customized_pipeline, request)
    return result


@router.post("/v1/chat/feedback")
def save_chat_feedback_to_db(request: FeedbackRequest) -> None:
    logger.info(f'fastrag feedback received.')
    # create mysql db instance
    mysql_db = MysqlDb()
    question, answer, feedback = request.question, request.answer, request.feedback
    feedback_str = 'dislike' if int(feedback) else 'like'
    logger.info(f'feedback question: [{question}], answer: [{answer}], feedback: [{feedback_str}]')
    # define sql statement
    sql = f"INSERT INTO feedback VALUES(null, '{question}', '{answer}', {feedback})"
    try:
        # execute sql statement and close db connection automatically
        mysql_db.insert(sql, None)
    except:
        # catch exceptions while inserting into db
        raise Exception("Exception occurred when inserting data into MySQL, please check the db session and your syntax.")
    else:
        logger.info('feedback inserted.')
        return "Succeed"

@router.post("/v1/retrieval/llmcache")
async def get_cache(request: QueryRequest):
    prompt = request.query
    from ..llmcache.cache import get
    result = get(prompt)
    print(result)
    if(result == None):
        print("cache miss >>>>>>>>>>>>>>>")
        response = RedirectResponse(url="/v1/chat/completions")
        return response
    else:
        print("cache hit >>>>>>>>>>>>>>>>")
        def stream_results():
            yield "data: {}\n\n".format(result['choices'][0]['text'])
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")

@router.post("/v1/chat/feedback")
def save_chat_response_to_db(request: FeedbackRequest) -> None:
    logger.info(f'fastrag feedback received.')
    mysql_db = MysqlDb()
    question, answer, feedback = request.question, request.answer, request.feedback
    feedback_str = 'dislike' if int(feedback) else 'like'
    logger.info(f'feedback question: [{question}], answer: [{answer}], feedback: [{feedback_str}]')
    sql = f"INSERT INTO feedback VALUES(null, '{question}', '{answer}', {feedback})"
    try:
        mysql_db.insert(sql, None)
    except:
        raise Exception("Exception occurred when inserting data into MySQL, please check the db session and your syntax.")
    else:
        logger.info('feedback inserted.')
        return "Succeed"

app.include_router(router)

@send_event_if_public_demo
def _process_dense_retrieval(request, result):
    text = result.replace('\n', '')
    #text = result['result'].replace('\n', '')
    print("=================", text)
    from ..llmcache.cache import put
    put(request.query, text)
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

    prediction = pipeline.run(query=request.query, params=params, debug=request.debug)
    # Ensure answers and documents exist, even if they're empty lists
    if not "documents" in prediction:
        prediction["documents"] = []
    if not "answers" in prediction:
        prediction["answers"] = []

    def stream_results():
        for _, result in enumerate(prediction["results"]):
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
    from ..llmcache.cache import put
    put(request.query, prediction["results"])
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
    from ..llmcache.cache import init_similar_cache_from_config, put
    if args.cache_chat_config_file:
        init_similar_cache_from_config(config_dir=args.cache_chat_config_file,
                                       embedding_model_dir=args.cache_embedding_model_dir)
        put("test","test")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
