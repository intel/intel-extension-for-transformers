<div align="center">

NeuralChat
===========================
<h3> A customizable framework to create your own LLM-driven AI apps within minutes</h3>

[🌟RESTful API](./docs/neuralchat_api.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻Examples](./examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Notebooks](./docs/full_notebooks.md)
</div>

# Introduction

NeuralChat is a powerful and flexible open framework that empowers you to effortlessly create LLM-centric AI applications, including chatbots and copilots.
* Support a range of hardware like [Intel Xeon Scalable processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html), [Intel Gaudi AI processors](https://habana.ai/products), [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html) and NVidia GPUs
* Leverage the leading AI frameworks (e.g., [PyTorch](https://pytorch.org/) and popular domain libraries (e.g., [Hugging Face](https://github.com/huggingface), [Langchain](https://www.langchain.com/)) with their extensions
* Support the model customizations through parameter-efficient fine-tuning, quantization, and sparsity. Released [Intel NeuralChat-7B LLM](https://huggingface.co/Intel/neural-chat-7b-v3-1), ranking #1 in Hugging Face leaderboard in Nov'23
* Provide a rich set of plugins that can augment the AI applications through retrieval-augmented generation (RAG) (e.g., [fastRAG](https://github.com/IntelLabs/fastRAG/tree/main)), content moderation, query caching, more
* Integrate with popular serving frameworks (e.g., [vLLM](https://github.com/vllm-project/vllm), [TGI](https://github.com/huggingface/text-generation-inference), [Triton](https://developer.nvidia.com/triton-inference-server)). Support [OpenAI](https://platform.openai.com/docs/introduction)-compatible API to simplify the creation or migration of AI applications

<a target="_blank" href="./docs/images/neuralchat_arch.png">
<p align="center">
  <img src="./docs/images/neuralchat_arch.png" alt="NeuralChat" width=600 height=340>
</p>
</a>

> NeuralChat is under active development. APIs are subject to change.

# Installation

NeuralChat is under Intel Extension for Transformers, so ensure the installation of Intel Extension for Transformers first by following the [installation](../../docs/installation.md). After that, install additional dependency for NeuralChat per your device:

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

NeuralChat provides OpenAI-compatible RESTful APIs for LLM inference, so you can use NeuralChat as a drop-in replacement for OpenAI APIs. NeuralChat service can also be accessible through [OpenAI client library](https://github.com/openai/openai-python), `curl` commands, and `requests` library. See [neuralchat_api.md](./docs/neuralchat_api.md).

### Launch OpenAI-compatible Service

NeuralChat launches a chatbot service using [Intel/neural-chat-7b-v3-1](https://huggingface.co/Intel/neural-chat-7b-v3-1) by default. You can customize the chatbot service by configuring the YAML file.

```shell
neuralchat_server start --config_file ./server/config/neuralchat.yaml
```

### Access the Service

Once the service is running, you can observe an OpenAI-compatible endpoint `/v1/chat/completions`. You can use any of below ways to access the endpoint.

#### Using OpenAI Client Library
```python
from openai import Client
# Replace 'your_api_key' with your actual OpenAI API key
api_key = 'your_api_key'
backend_url = 'http://127.0.0.1:80/v1/chat/completions'
client = Client(api_key=api_key, base_url=backend_url)
response = client.ChatCompletion.create(
      model="Intel/neural-chat-7b-v3-1",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."},
      ]
)
print(response)
```

#### Using Curl
```shell
curl http://127.0.0.1:80/v1/chat/completions \
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

## Langchain Extension APIs

Intel Extension for Transformers provides a comprehensive suite of Langchain-based extension APIs, including advanced retrievers, embedding models, and vector stores. These enhancements are carefully crafted to expand the capabilities of the original langchain API, ultimately boosting overall performance. This extension is specifically tailored to enhance the functionality and performance of RAG.

### Vector Stores

We introduce enhanced vector store operations, enabling users to adjust and fine-tune their settings even after the chatbot has been initialized, offering a more adaptable and user-friendly experience. For langchain users, integrating and utilizing optimized Vector Stores is straightforward by replacing the original Chroma API in langchain.

```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from intel_extension_for_transformers.langchain.vectorstores import Chroma
retriever = VectorStoreRetriever(vectorstore=Chroma(...))
retrievalQA = RetrievalQA.from_llm(llm=HuggingFacePipeline(...), retriever=retriever)
```

### Retrievers

We provide optimized retrievers such as `VectorStoreRetriever`, `ChildParentRetriever` to efficiently handle vectorstore operations, ensuring optimal retrieval performance.

```python
from intel_extension_for_transformers.langchain.retrievers import ChildParentRetriever
from langchain.vectorstores import Chroma
retriever = ChildParentRetriever(vectorstore=Chroma(documents=child_documents), parentstore=Chroma(documents=parent_documents), search_type=xxx, search_kwargs={...})
docs=retriever.get_relevant_documents("Intel")
```

Please refer to this [documentation](./pipeline/plugins/retrieval/README.md) for more details.


## Advanced Features

NeuralChat introduces `plugins` that offer a wide range of useful LLM utilities and features, enhancing the capabilities of the chatbot. Additionally, NeuralChat provides advanced model optimization technologies such as `Automatic Mixed Precision (AMP)` and `Weight Only Quantization`. These technologies enable users to run a high-throughput chatbot efficiently. NeuralChat further supports fine-tuning the pretrained LLMs for tasks such as text generation, summarization, code generation, and even Text-to-Speech (TTS) models, allowing users to create customized chatbots tailored to their specific needs.

Please refer to this [documentation](./docs/advanced_features.md) for more details.

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

We provide Jupyter notebooks to help users explore how to create, deploy, and customize chatbots on different hardware architecture. The selected notebooks are shown below:

| Notebook | Title                                       | Description                                                | Link                                           |
| ------- | --------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| #1     | Getting Started on Intel CPU SPR          | Learn how to create chatbot on SPR                      | [Notebook](./docs/notebooks/build_chatbot_on_spr.ipynb) |
| #2     | Getting Started on Habana Gaudi1/Gaudi2   | Learn how to create chatbot on Habana Gaudi1/Gaudi2     | [Notebook](./docs/notebooks/build_chatbot_on_habana_gaudi.ipynb) |
| #3     | Deploying Chatbot on Intel CPU SPR        | Learn how to deploy chatbot on SPR                      | [Notebook](./docs/notebooks/deploy_chatbot_on_spr.ipynb) |
| #4     | Deploying Chatbot on Habana Gaudi1/Gaudi2 | Learn how to deploy chatbot on Habana Gaudi1/Gaudi2     | [Notebook](./docs/notebooks/deploy_chatbot_on_habana_gaudi.ipynb) |
| #5     | Deploying Chatbot with Load Balance       | Learn how to deploy chatbot with load balance on SPR    | [Notebook](./docs/notebooks/chatbot_with_load_balance.ipynb) |


🌟Please refer to [HERE](docs/full_notebooks.md) for the full notebooks.
