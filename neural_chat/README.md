**NeuralChat** is a powerful and versatile chatbot designed to facilitate textual or voice conversations. By providing NeuralChat with textual or voice instructions, users can receive accurate and relevant responses. We provide a comprehensive API for building a highly customizable end-to-end chatbot service, covering model pre-training, model fine-tuning, model compression, prompt engineering, knowledge base retrieval, talkingbot and quick deployment.

# Installation

NeuralChat is part of Intel extension for transformers, so users just need to install 'intel-extension-for-transformers'.

## Install from Pypi
```bash
pip install intel-extension-for-transformers
```
> For more installation method, please refer to [Installation Page](../docs/installation.md)

<a name="quickstart"></a>
# Quick Start

Users can have a try of NeuralChat with [NeuralChat Command Line](./cli/README.md) or Python API.


## Text Chat


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


**command line experience**

```shell
neuralchat voicechat --input assets/audio/say_hello.wav --output response.wav
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
    def chatbot_callback(prompt):
        response = chatbot.predict(prompt)
        return response
    server_executor = NeuralChatServerExecutor()
    server_executor.register_callback(chatbot_callback)
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
    
    ```

**Access Voice Chat Service**

```shell
neuralchat_client voicechat --server_ip 127.0.0.1 --port 8000 --input say_hello.wav --output response.wav
```

**Access Retrieval Service**
```shell
neuralchat_client retrieval --server_ip 127.0.0.1 --port 8000 --input ./docs/
```

