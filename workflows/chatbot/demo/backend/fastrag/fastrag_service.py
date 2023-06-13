from haystack.document_stores import InMemoryDocumentStore,ElasticsearchDocumentStore

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi import APIRouter
import sys
import platform
import os
import warnings
import uvicorn
import argparse
import collections
import json
import logging
import time
from typing import Any, Dict
from haystack.telemetry import send_event_if_public_demo



inc_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
es_document_store = ElasticsearchDocumentStore(host="localhost", index="elastic_index_1",
                                               port=9200, search_fields=["content", "title"])

LOGDIR = "."
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                encoded_message = line.encode("utf-8", "ignore").decode("utf-8")
                self.logger.log(self.log_level, encoded_message.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            encoded_message = self.linebuf.encode("utf-8", "ignore").decode("utf-8")
            self.logger.log(self.log_level, encoded_message.rstrip())
        self.linebuf = ""


handler = None

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        if sys.version_info[1] >= 9:
            # This is for windows
            logging.basicConfig(level=logging.INFO, encoding="utf-8")
        else:
            if platform.system() == "Windows":
                warnings.warn("If you are running on Windows, "
                              "we recommend you use Python >= 3.9 for UTF-8 encoding.")
            logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True
        )
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger

logger = build_logger("fastrag_service", f"fastrag_service.log")

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=80)
parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
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
inc_retriever = BM25Retriever(document_store=inc_document_store, top_k = 10)
es_retriever = BM25Retriever(document_store=es_document_store, top_k = 10)

reranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32, top_k=5)

#pmodel = FastRAGPromptModel(model_name_or_path="../llama-7b-hf")
pmodel = FastRAGPromptModel(model_name_or_path=args.model_path)

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
    prompt_text="Given the context please answer the question. Context: $documents; Question: $query; Answer:",
)
prompt.add_prompt_template(custom_template)
prompt.default_prompt_template = custom_template
inc_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])
wiki_pipeline.add_node(component=prompt, name="Prompter", inputs=["Shaper"])



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
    articles: Optional[List[str]] = []
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
    logger.info("fastrag query received for {domain}")
    if domain == "INC":
        result = _process_request(inc_pipeline, request)
    elif domain == "WIKI":
        result = _process_request(wiki_pipeline, request)
    elif domain == "Customized":
        customized_document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
        customized_documents = []
        for i, d in enumerate(request.articles):
            customized_documents.append(Document(d, id=i))

        customized_document_store.write_documents(customized_documents)
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

