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
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, GenerationConfig
from logger import build_logger
from haystack.nodes import MarkdownConverter, TextConverter
from utils import detect_language
from starlette.responses import RedirectResponse

logger = build_logger("fastrag_service", f"fastrag_service.log")
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=80)
parser.add_argument("--model-path", type=str, default="mosaicml/mpt-7b-chat")
parser.add_argument(
    "--cache-chat-config-file", default="cache_config.yml", help="the cache config file"
)
parser.add_argument(
    "--cache-embedding-model-dir", default="./instructor-large", help="the cache embedding model directory"
)
args = parser.parse_args()
logger.info(f"args: {args}")

def ask_gm_documents_dense_embedding(folder_path, process_content=False):
    documents = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".json"):
                documents = doc_index.d_load_jsonl_file(os.path.join(dirpath, filename), process_content, documents)
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

reranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32, top_k=2)

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
langchain_tok.pad_token = langchain_tok.eos_token
langchain_tok.add_special_tokens({'pad_token': '[PAD]'})
if not os.path.exists("/tmp/young_pat_dense_retrieval"):
    from young_pat import d_load_xlsx, persist_embedding
    documents = d_load_xlsx("./doc/young_pat/pat.xlsx", True)
    persist_embedding(documents, "/tmp/young_pat_dense_retrieval", model_path="hkunlp/instructor-large")
langchain_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
langchain_vectordb = Chroma(persist_directory="/tmp/young_pat_dense_retrieval", embedding_function=langchain_embeddings)
langchain_retriever = langchain_vectordb.as_retriever(search_type = "mmr", search_kwargs = {"k": 1, "fetch_k": 5})

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
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

@router.post("/v1/chat/completions")
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
            retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs = {"k": 2, "fetch_k": 5})
            prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:"""
            from langchain.prompts import PromptTemplate
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain_type_kwargs = {"prompt": PROMPT}
            start_time = time.time()
            qa = RetrievalQA.from_chain_type(llm=langchain_model, retriever=retriever, chain_type="stuff",
                                     chain_type_kwargs=chain_type_kwargs)
            chain_result = qa.run(request.query)
            end_time = time.time()
            print("inference cost {} seconds.".format(end_time - start_time))
            print(chain_result)
            result = _process_dense_retrieval(request, chain_result)
        elif request.embedding == 'sparse':
            askgm_document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_askgm_sparse", port=9200, search_fields=["content", "title"])
            askgm_pipeline = Pipeline()
            askgm_retriever = BM25Retriever(document_store=askgm_document_store, top_k = 10)
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
            logger.info("Sparse Retrieval for ASK_GM done...")
            result = _process_request(askgm_pipeline, request)
        elif domain == "Young_Pat":
            if request.embedding == "dense":
                docs = langchain_retriever.get_relevant_documents(request.query)
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
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=0,
                    min_new_tokens=1,
                    max_new_tokens=256,
                    temperature=0.001,
                    top_p=0.9,
                    top_k=3,
                    repetition_penalty=1.1,
                    do_sample=True,
                    stopping_criteria=StoppingCriteriaList([stop])
                )
                generation_config = GenerationConfig(**generate_kwargs)
                start_time = time.time()
                with torch.no_grad():
                    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                        output = langchain_model.generate(**inputs, generation_config=generation_config)
                    output = output[0][inputs["input_ids"].shape[-1]:]
                generated_texts = langchain_tok.decode(output, skip_special_tokens=True)
                end_time = time.time()
                print("inference cost {} seconds.".format(end_time - start_time))
                print(generated_texts)
                result = _process_dense_retrieval(generated_texts)
            else:
                customized_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
                from young_pat import s_load_xlsx
                customized_document_store = s_load_xlsx("./doc/young_pat/pat.xlsx", True, customized_document_store)
                young_pat_pipeline = Pipeline()
                young_pat_retriever = BM25Retriever(document_store=customized_document_store, top_k = 10)
                young_pat_pipeline.add_node(component=young_pat_retriever, name="Retriever", inputs=["Query"])
                young_pat_pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
                young_pat_pipeline.add_node(component=shaper, name="Shaper", inputs=["Reranker"])
                from haystack.nodes.prompt.prompt_node import PromptNode
                from haystack.nodes.prompt.prompt_node import PromptTemplate
                prompt = PromptNode(model_name_or_path = pmodel)
                custom_template = PromptTemplate(
                            name="llama-instruct",
                            prompt_text="Given the context please answer the question. Context: $documents; Question: $query; Answer:")
                prompt.add_prompt_template(custom_template)
                prompt.default_prompt_template = custom_template
                young_pat_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])
                logger.info("Sparse Retrieval for Young_Pat done...")
                result = _process_request(young_pat_pipeline, request)
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
            yield "data: Response from Cache: {}\n\n".format(result['choices'][0]['text'])
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")

app.include_router(router)

@send_event_if_public_demo
def _process_dense_retrieval(request, result):
    text = result['output_text'].replace('\n', '')
    #text = result['result'].replace('\n', '')
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

    prediction  = pipeline.run(query=request.query, params=params, debug=request.debug)
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

