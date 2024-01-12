<div align="center">

NeuralChat
===========================
<h3> A customizable chatbot framework to create your own chatbot within minutes</h3>

[😃API](./docs/neuralchat_api.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻Examples](./examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Notebooks](./docs/full_notebooks.md)
</div>

# Introduction

NeuralChat is a customizable chat framework designed to easily create user own chatbot that can be efficiently deployed across multiple architectures (e.g., Intel® Xeon® Scalable processors, Habana® Gaudi® AI processors). NeuralChat is built on top of large language models (LLMs) and provides a set of strong capabilities including LLM fine-tuning, optimization, and inference, together with a rich set of plugins such as knowledge retrieval, query caching, etc. With NeuralChat, you can easily create a text-based or audio-based chatbot within minutes and deploy on user favorite platform rapidly.

<a target="_blank" href="./docs/images/neuralchat_arch.png">
<p align="center">
  <img src="./docs/images/neuralchat_arch.png" alt="NeuralChat" width=800 height=612>
</p>
</a>

> NeuralChat is under active development with some experimental features (APIs are subject to change).

# Installation

NeuralChat is seamlessly integrated into the Intel Extension for Transformers. Please refer to [Installation](../../docs/installation.md) page for step by step instructions.

Once you've installed the Intel Extension for Transformers, install NeuralChat's dependencies based on your device:
```shell
# For CPU device
pip install -r requirements_cpu.txt

# For HPU device
pip install -r requirements_hpu.txt

# For XPU device
pip install -r requirements_xpu.txt

# For CUDA
pip install -r requirements.txt
```

# Getting Started

## OpenAI-Compatible RESTful APIs

NeuralChat provides OpenAI-compatible APIs for LLM inference, so you can use NeuralChat as a local drop-in replacement for OpenAI APIs. The NeuralChat server is compatible with both [openai-python library](https://github.com/openai/openai-python) and cURL commands. See [neuralchat_api.md](./docs/neuralchat_api.md).


### Launch Service

NeuralChat defaults to running the "Intel/neural-chat-7b-v3-1" model, and you can customize the chatbot service by configuring the YAML file.

```shell
neuralchat_server start --config_file ./server/config/neuralchat.yaml
```

### Access Service

Once the service is running, you can use the OpenAI-compatible endpoint `/v1/chat/completions` by doing requests. You can use below ways to query the endpoints.

#### Using Curl
```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Intel/neural-chat-7b-v3-1",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."}
    ]
    }'
```

#### Using Python Requests Library

```python
# Python code
import requests
url = 'http://127.0.0.1:80/v1/chat/completions'
headers = {'Content-Type': 'application/json'}
data = '{"model": "Intel/neural-chat-7b-v3-1", "messages": [ \
          {"role": "system", "content": "You are a helpful assistant."}, \
          {"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."}] \
       }'
response = requests.post(url, headers=headers, data=data)
print(response.json())
```

#### Using OpenAI Client Library
```python
# Python code
from openai import Client
# Replace 'your_api_key' with your actual OpenAI API key
api_key = 'your_api_key'
backend_url = 'http://127.0.0.1:80/v1/chat/completions'
client = Client(api_key=api_key, base_url=backend_url)
chat_response = client.ChatCompletion.create(
      model="Intel/neural-chat-7b-v3-1",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."},
      ]
)
print(chat_response)
```

## Langchain Extension APIs

ITREX serves not only as an Intel extension for transformers but also as a langchain extension. We provide a comprehensive suite of langchain-based extension APIs, including advanced retrievers, embedding models, and vector stores. These enhancements are carefully crafted to expand the capabilities of the original langchain API, ultimately boosting overall performance. This extension is specifically tailored to enhance the functionality and performance of RAG (Retrieval-Augmented Generation).

### Vector Stores

We introduce enhanced vector store operations, enabling users to adjust and fine-tune their settings even after the chatbot has been initialized, offering a more adaptable and user-friendly experience. For langchain users, integrating and utilizing optimized Vector Stores is straightforward by replacing the original Chroma API in langchain.

```python
# Python code
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from intel_extension_for_transformers.langchain.vectorstores import Chroma
retriever = VectorStoreRetriever(vectorstore=Chroma(...))
retrievalQA = RetrievalQA.from_llm(llm=HuggingFacePipeline(...), retriever=retriever)
```

### Retrievers

We provide some optimized retrievers such as `VectorStoreRetriever`, `ChildParentRetriever` to efficiently handle vectorstore operations, ensuring optimal retrieval performance.

```python
# Python code
from intel_extension_for_transformers.langchain.retrievers import ChildParentRetriever
from langchain.vectorstores import Chroma
retriever = ChildParentRetriever(vectorstore=Chroma(documents=child_documents), parentstore=Chroma(documents=parent_documents), search_type=xxx, search_kwargs={...})
docs=retriever.get_relevant_documents("Intel")
```

The details please refer to this [documentation](./pipeline/plugins/retrieval/README.md).


## Advanced Features

NeuralChat introduces `plugins` that offer a wide array of useful LLM utilities and features, enhancing the capabilities of the chatbot. Additionally, NeuralChat provides advanced model optimization technologies such as `Automatic Mixed Precision (AMP)` and `Weight Only Quantization`. These technologies enable users to run a high-throughput chatbot efficiently. NeuralChat further supports fine-tuning the pretrained large language model (LLM) for tasks such as text generation, summarization, code generation, and even Text-to-Speech (TTS) models, allowing users to create customized chatbots tailored to their specific needs.

The details please refer to this [documentation](./docs/advanced_features.md).

# Models

## Supported  Models
The table below displays the validated model list in NeuralChat for both inference and fine-tuning.
|Pretrained model| Text Generation (Completions) | Text Generation (Chat Completions) | Summarization | Code Generation | 
|------------------------------------|:---:|:---:|:---:|:---:|
|Intel/neural-chat-7b-v1-1| ✅| ✅| ✅| ✅    |
|Intel/neural-chat-7b-v3-1| ✅| ✅| ✅| ✅    |
|LLaMA series| ✅| ✅|✅| ✅    |
|LLaMA2 series| ✅| ✅|✅| ✅    |
|GPT-J| ✅| ✅|✅| ✅    |
|MPT series| ✅| ✅|✅| ✅    |
|Mistral series| ✅| ✅|✅| ✅    |
|Mixtral series| ✅| ✅|✅| ✅    |
|SOLAR Series| ✅| ✅|✅| ✅    |
|ChatGLM series| ✅| ✅|✅| ✅    |
|Qwen series| ✅| ✅|✅| ✅    |
|StarCoder series|   |   |   | ✅ |
|CodeLLaMA series|   |   |   | ✅ |
|CodeGen series|   |   |   | ✅ |
|MagicCoder series|   |   |   | ✅ |


# Notebooks

Welcome to use Jupyter notebooks to explore how to create, deploy, and customize chatbots on multiple architectures, including Intel Xeon Scalable Processors, Intel Gaudi2, Intel Xeon CPU Max Series, Intel Data Center GPU Max Series, Intel Arc Series, and Intel Core Processors, and others. The selected notebooks are shown below:

| Notebook | Title                                       | Description                                                | Link                                           |
| ------- | --------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| #1     | Getting Started on Intel CPU SPR          | Learn how to create chatbot on SPR                      | [Notebook](./docs/notebooks/build_chatbot_on_spr.ipynb) |
| #2     | Getting Started on Habana Gaudi1/Gaudi2   | Learn how to create chatbot on Habana Gaudi1/Gaudi2     | [Notebook](./docs/notebooks/build_chatbot_on_habana_gaudi.ipynb) |
| #3     | Deploying Chatbot on Intel CPU SPR        | Learn how to deploy chatbot on SPR                      | [Notebook](./docs/notebooks/deploy_chatbot_on_spr.ipynb) |
| #4     | Deploying Chatbot on Habana Gaudi1/Gaudi2 | Learn how to deploy chatbot on Habana Gaudi1/Gaudi2     | [Notebook](./docs/notebooks/deploy_chatbot_on_habana_gaudi.ipynb) |
| #5     | Deploying Chatbot with Load Balance       | Learn how to deploy chatbot with load balance on SPR    | [Notebook](./docs/notebooks/chatbot_with_load_balance.ipynb) |


🌟Please refer to [HERE](docs/full_notebooks.md) for the full notebooks.
