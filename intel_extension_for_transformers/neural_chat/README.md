<div align="center">

NeuralChat
===========================
<h3> A customizable chatbot framework that empowers users to create/personalize user own chatbot on any device within minutes.</h3>

---
<div align="left">

## Content

1. [Introduction](#introduction)

2. [Installation](#installation)

3. [Getting Started](#getting-started)

    3.1 [Local Mode](#local-mode)

    3.2 [Server Mode](#server-mode)

    3.2.1 [Launch Server](#launch-server)

    3.2.2 [Access Server](#access-server)

4. [Advanced Topics](#advanced-topics)

5. [Jupyter Notebooks](#jupyter-notebooks)

## Introduction

NeuralChat is a customizable chat framework designed to easily create user own chatbot that can be efficiently deployed across multiple architectures (e.g., Intel Xeon Scalable processor, Habana Gaudi AI processors and Nvidia Data Center GPUs). NeuralChat is built on top of large language models (LLMs) and provides a set of strong capabilities including LLM fine-tuning, optimization, and inference, together with a rich set of plugins such as knowledge retrieval, response caching, etc. With NeuralChat, you can easily create a text-based or audio-based chatbot within minutes and deploy on user favorite platform rapidly.

<a target="_blank" href="./assets/pictures/neuralchat.png">
<p align="center">
  <img src="./assets/pictures/neuralchat.png" alt="NeuralChat" width=600 height=250>
</p>
</a>

> NeuralChat is under active development with some experimental features (APIs are subject to change).

## Installation

NeuralChat is seamlessly integrated into the Intel Extension for Transformers. Please refer to [Installation](../docs/installation.md) page for step by step instructions.

## Getting Started

NeuralChat could be deployed as local mode and server mode.

### Local Mode

NeuralChat is deployed at local machine after installation, user can access it through:

command line

```shell
neuralchat predict --query "Tell me about Intel Xeon Scalable Processors."
```

or python code

```python
>>> from intel_extension_for_transformers.neural_chat import build_chatbot
>>> chatbot = build_chatbot()
>>> response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
```

### Server Mode

NeuralChat is deployed at remote machine as a service, user can access it through curl with Restful API.

#### Launch Server

Executing below cmd starts a chatbot server for client access:

```shell
neuralchat_server start --config_file ./server/config/neuralchat.yaml
```

#### Access Server

Using `curl` command like below to post request to the launched chatbot server.

```shell
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Tell me about Intel Xeon Scalable Processors."}' http://127.0.0.1:80/v1/chat/completions
```

## Advanced Topics

### Plugins

NeuralChat introduces the `plugin` machenism which integrate a lot of useful LLM utils and features to augment the chatbot's capability. Such plugins are applied in the chatbot pipeline for inference.

Belows are supported plugins.

- [knowledge retrieval](./pipeline/plugins/retrievers/)

    Knowledge retrieval consists of document indexing for efficient retrieval of relevant information, including Dense Indexing based on LangChain and Sparse Indexing based on fastRAG, document rankers to prioritize the most relevant responses.

- [query caching](./pipeline/plugins/caching/)

    Query caching enables the fast path to get the response without LLM inference and therefore improves the chat response time

- [prompt optimization](./pipeline/plugins/prompts/)

    Prompt optimization supports auto prompt engineering to improve user prompts.

- [memory controller](./pipeline/plugins/memory/)

    Memory controller enables the efficient memory utilization.

- [safety checker](./pipeline/plugins/security/)

    Safety checker enables the sensitive content check on inputs and outputs of the chatbot.

User could enable, disable, and even change the default behavior of all supported plugins like below

```python
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig, plugins

plugins.retrieval.enable = True
plugins.retrieval.args["input_path"]="./assets/docs/"
conf = PipelineConf(plugins=plugins)
chatbot = build_chatbot(conf)

```

### Finetune

NeuralChat supports fine-tuning the pretrained large language model (LLM) for text-generation, summarization, code generation tasks, and even TTS model, for user to create the customized chatbot.

```shell
neuralchat finetune --base_model "meta-llama/Llama-2-7b-chat-hf" --config pipeline/finetuning/config/finetuning.yaml
```

or

```python
>>> from intel_extension_for_transformers.neural_chat import finetune_model, TextGenerationFinetuningConfig
>>> finetune_cfg = TextGenerationFinetuningConfig() # change to SummarizationFinetuningConfig or CodeGenerationFinetuningConfig if fine-tuning for summarization and code generation tasks
>>> finetuned_model = finetune_model(finetune_cfg)
```

### Optimization

NeuralChat provides several model optimization technologies, like `AMP(advanced mixed precision)` and `WeightOnly Quantization`, to allow user to define a customized chatbot.

```shell
neuralchat optimize --base_model "meta-llama/Llama-2-7b-chat-hf" --config pipeline/optimization/config/optimization.yaml
```

or

```python
>>> from intel_extension_for_transformers.neural_chat import build_chatbot, AMPConfig
>>> pipeline_cfg = PipelineConfig(optimization_config=AMPConfig())
>>> chatbot = build_chatbot(pipeline_cfg)
```

## Jupyter Notebooks 

Check out the latest notebooks to know how to build and customize a chatbot on different platforms.

| **Notebook** | **Description** |
| :----------: | :-------------: |
| [build chatbot on Intel Xeon Platforms](./docs/notebooks/chatbot_on_intel_cpu.ipynb) | create a chatbot on Intel Xeon Platforms|
| [build chatbot on Intel Habana Platforms](./docs/notebooks/chatbot_on_intel_habana_gpu.ipynb) | create a chatbot on Intel Habana Platforms|
| [build chatbot on Nvidia GPU Platforms](./docs/notebooks/chatbot_on_nv_gpu.ipynb) | create a chatbot on Nvidia GPU Platforms|
| [finetune on Nvidia GPU Platforms](./examples/instruction_tuning/finetune_on_Nvidia_GPU.ipynb) | finetuning LLaMA2 and MPT on Nvidia GPU Platforms|

