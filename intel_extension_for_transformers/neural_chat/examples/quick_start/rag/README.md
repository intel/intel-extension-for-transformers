
# Build RAG (retriveval augment generation) example with Intel® Extension for Transformers neural-chat on Intel GPU

#1. Setup Environment

## prereqeust
For GPU, oneAPI 2024.0 is required

## 1.1 Install intel-extension-for-transformers

```
conda create -n itrex-rag python=3.9
conda activate itrex-rag 

source /opt/intel/oneapi/setvars.sh

git clone https://github.com/intel/intel-extension-for-transformers.git ~/itrex
cp requirements-gpu.patch ~/itrex
cd ~/itrex
git checkout v1.4rc1

patch -p1  < requirements-gpu.patch

pip install -r requirements-gpu.txt
pip install -v .
```


```
pip install accelerate
pip install transformers_stream_generator

cd ~/itrex/intel_extension_for_transformers/neural_chat
```

To setup CPU platform go to 1.2.1
To setup GPU platform go to 1.2.2

### 1.2.1 CPU Platform
`pip install -r requirements_cpu.txt`

Got to 1.3

### 1.2.2 Intel GPU Platform

### prereqeust
I suppose you install GPU driver and oneAPI already.
If NOT, please setup you GPU platform and oneAPI 2024.0

`pip install -r requirements_xpu.txt`

Got to 1.3

## 1.3 Install retrieval dependency

```
cd ~/itrex/intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval
pip install -r requirements.txt
pip install -U langchain-community
```

# 2. Run the RAG in command mode

## Usage

Exagmple 1. Run example

```
python retrieval.py
```

Exagmple 2. Run example disable retrieval

```
python retrieval.py --no-retrieval
```

# 3. Run RAG in client server mode 

# 3.1 Start the service

python neural_ser.py

Here is the completely output:
```
/home/xiguiwang/ws2/conda/itrex-rag/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: ''If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
Namespace(config='./neuralchat.yaml')
2024-03-18 11:05:35,388 - datasets - INFO - PyTorch version 2.1.0a0+cxx11.abi available.
/home/xiguiwang/ws2/conda/itrex-rag/lib/python3.9/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/home/xiguiwang/ws2/conda/itrex-rag/lib/python3.9/site-packages/langchain/document_loaders/__init__.py:36: LangChainDeprecationWarning: Importing document loaders from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:

`from langchain_community.document_loaders import UnstructuredMarkdownLoader`.

To install langchain-community run `pip install -U langchain-community`.
  warnings.warn(
/home/xiguiwang/ws2/conda/itrex-rag/lib/python3.9/site-packages/langchain/retrievers/__init__.py:46: LangChainDeprecationWarning: Importing this retriever from langchain is deprecated. Importing it from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:

`from langchain_community.retrievers import BM25Retriever`.

To install langchain-community run `pip install -U langchain-community`.
  warnings.warn(
/home/xiguiwang/ws2/conda/itrex-rag/lib/python3.9/site-packages/langchain/embeddings/__init__.py:29: LangChainDeprecationWarning: Importing embeddings from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:

`from langchain_community.embeddings import GooglePalmEmbeddings`.

To install langchain-community run `pip install -U langchain-community`.
  warnings.warn(
create retrieval plugin instance...
plugin parameters:  {'retrieval_type': 'default', 'input_path': './text/', 'embedding_model': 'BAAI/bge-base-en-v1.5', 'persist_dir': './output', 'max_length': 512, 'process': False}
2024-03-18 11:05:37,228 - sentence_transformers.SentenceTransformer - INFO - Load pretrained SentenceTransformer: BAAI/bge-base-en-v1.5
2024-03-18 11:05:42,318 - sentence_transformers.SentenceTransformer - INFO - Use pytorch device_name: cpu
2024-03-18 11:05:42,331 - easyocr.easyocr - WARNING - Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU.
2024-03-18 11:05:44,677 - easyocr.easyocr - WARNING - Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU.
2024-03-18 11:06:30,726 - root - INFO - The parsing for the uploaded files is finished.
2024-03-18 11:06:30,726 - root - INFO - The format of parsed documents is transferred.
2024-03-18 11:06:30,733 - chromadb.telemetry.product.posthog - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.09s/it]
2024-03-18 11:06:32,300 - root - ERROR - The chosen retrieval type remains outside the supported scope.
2024-03-18 11:06:32,300 - root - INFO - The retriever is successfully built.
Loading model Intel/neural-chat-7b-v3-1
model-00001-of-00002.safetensors: 100%|███████████████████████████████████████████████████████████████████████| 9.94G/9.94G [12:37<00:00, 13.1MB/s]
Downloading shards:  50%|█████████████████████████████████████████████▌                                             | 1/2 [12:39<12:39, 759.82s/it]
model-00002-of-00002.safetensors: 100%|███████████████████████████████████████████████████████████████████████| 4.54G/4.54G [05:45<00:00, 13.2MB/s]
Downloading shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2/2 [18:27<00:00, 553.63s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.07s/it]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████| 111/111 [00:00<00:00, 48.5kB/s]
2024-03-18 11:25:09,803 - root - INFO - Model loaded.
Loading config settings from the environment...
INFO:     Started server process [1520808]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:54544 - "POST /v1/chat/completions HTTP/1.1" 422 Unprocessable Entity
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.42it/s]
2024-03-18 11:26:17,241 - root - INFO - Chat with QA Agent.
INFO:     127.0.0.1:42680 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO:     127.0.0.1:38610 - "POST /v1/models HTTP/1.1" 200 OK
INFO:     127.0.0.1:53712 - "POST /v1/models HTTP/1.1" 200 OK
INFO:     127.0.0.1:38518 - "POST /v1/models HTTP/1.1" 200 OK
INFO:     127.0.0.1:34442 - "POST /v1/models HTTP/1.1" 200 OK
INFO:     127.0.0.1:60016 - "POST /v1/chat/completions HTTP/1.1" 200 OK
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.41it/s]
2024-03-18 11:44:01,205 - root - INFO - Chat with QA Agent.

```

# 3.1.1 verify the connection to serivce is OK.


`curl -vv -X POST http://127.0.0.1:8000/v1/chat/completions`

Make sure there is no network connection and proxy setting issue 

# 3.1.2 

Sent a request to neural-chat server

```
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Intel/neural-chat-7b-v3-1",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."}
    ]
    }'
```

# 3.2 Set up Server mode UI

Create UI conda envitonment
```
conda create -n chatbot-ui python=3.9
conda activate chatbot-ui

cd itrex/intel_extension_for_transformers/neural_chat/ui/gradio/basic
pip install -r requirements.txt

pip install gradio==3.36.0
pip install pydantic==1.10.13
```

# 3.3 start the web serice

In directory, `itrex/intel_extension_for_transformers/neural_chat/ui/gradio/basic`, start app.py

```
python app.py
```
