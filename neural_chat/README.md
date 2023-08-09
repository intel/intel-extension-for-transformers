**NeuralChat** is a powerful and versatile chatbot designed to facilitate textual or voice conversations. By providing NeuralChat with textual or voice instructions, users can receive accurate and relevant responses. We provide a comprehensive API for building a highly customizable end-to-end chatbot service, covering model pre-training, model fine-tuning, model compression, prompt engineering, knowledge base retrieval, talkingbot and quick deployment.

# Installation

NeuralChat is seamlessly integrated into the Intel Extension for Transformers. Getting started is quick and simple, just simply install 'intel-extension-for-transformers'.

## Install from Pypi
```bash
pip install intel-extension-for-transformers
```
> For more installation method, please refer to [Installation Page](../docs/installation.md)

<a name="quickstart"></a>
# Quick Start

Users can have a try of NeuralChat with [NeuralChat Command Line](./cli/README.md) or Python API.


## Text Chat

Giving NeuralChat the textual instruction, it will respond with the textual response.

**command line experience**

```shell
neuralchat textchat --prompt "Tell me about Intel Xeon processors."
```

**Python API experience**

```python
>>> from neural_chat.config import NeuralChatConfig
>>> from neural_chat.chatbot import NeuralChatBot
>>> config = NeuralChatConfig()
>>> chatbot = NeuralChatBot(config)
>>> chatbot.build_chatbot()
>>> response = chatbot.predict("Tell me about Intel Xeon processors.")
```


## Voice Chat

In the context of voice chat, users have the option to engage in various modes: utilizing input audio and receiving output audio, employing input audio and receiving textual output, or providing input in textual form and receiving audio output.

**command line experience**

- audio in and audio output
    ```shell
    neuralchat voicechat --input assets/audio/say_hello.wav --output response.wav
    ```

- audio in and text output
    ```shell
    neuralchat voicechat --input assets/audio/say_hello.wav
    ```

- text in and audio output
    ```shell
    neuralchat voicechat --input "Tell me about Intel Xeon processors." --output response.wav
    ```


**Python API experience**

For the Python API code, users have the option to enable different voice chat modes by setting audio_input to True for input or audio_output to True for output.

```python
>>> from neural_chat.config import NeuralChatConfig
>>> from neural_chat.chatbot import NeuralChatBot
>>> config = NeuralChatConfig(audio_input=True, audio_output=True)
>>> chatbot = NeuralChatBot(config)
>>> chatbot.build_chatbot()
>>> result = chatbot.predict("Tell me about Intel Xeon processors.")
```

## Finetuning

Finetune the pretrained large language model (LLM) with the instruction-following dataset for creating the customized chatbot is very easy for NeuralChat.

**command line experience**

```shell
neuralchat finetune --base_model "meta-llama/Llama-2-7b-chat-hf" --config finetuning/config/finetuning.config
```


**Python API experience**

```python
>>> from neural_chat.config import FinetuningConfig, NeuralChatConfig
>>> from neural_chat.chatbot import NeuralChatBot
>>> finetuneCfg = FinetuningConfig()
>>> config = NeuralChatConfig(finetuneConfig=finetuneCfg)
>>> chatbot = NeuralChatBot(config)
>>> chatbot.build_chatbot()
>>> chatbot.finetune()
>>> response = chatbot.predict("Tell me about Intel Xeon processors.")
```


<a name="quickstartserver"></a>
# Quick Start Server

Users can have a try of NeuralChat server with [NeuralChat Server Command Line](./server/README.md).


**Start Server**
- Command Line (Recommended)
    ```shell
    neuralchat_server start --config_file ./conf/neuralchat.yaml
    ```

- Python API
    ```python
    from neuralchat.server.neuralchat_server import NeuralChatServerExecutor
    server_executor = NeuralChatServerExecutor()
    server_executor(
        config_file="./conf/neuralchat.yaml", 
        log_file="./log/neuralchat.log")
    ```

**Access Text Chat Service**

- Command Line
    ```shell
    neuralchat_client textchat --server_ip 127.0.0.1 --port 8000 --prompt "Tell me about Intel Xeon processors."
    ```

- Python API
    ```python
    from neuralchat.server.neuralchat_client import TextChatClientExecutor

    executor = TextChatClientExecutor()
    executor(
        prompt="Tell me about Intel Xeon processors.",
        server_ip="127.0.0.1",
        port=8000)
    ```

- Curl
    ```
    curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Tell me about Intel Xeon processors."}' http://127.0.0.1:80/v1/chat/completions
    ```

**Access Voice Chat Service**

```shell
neuralchat_client voicechat --server_ip 127.0.0.1 --port 8000 --input say_hello.wav --output response.wav
```

**Access Retrieval Service**
```shell
neuralchat_client retrieval --server_ip 127.0.0.1 --port 8000 --input ./docs/
```

