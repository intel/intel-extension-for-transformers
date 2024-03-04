# Start NeuralChat Text Generation Service with Docker
Intel Neural Chat Text Generation Dockerfile installer for Ubuntu22.04/Habana Gaudi.

Following the instruction of this README.html, you will start a Text Generation HTTP service with NeuralChat. The whole procedure is very clear and easy for customers to use with only two docker commands.The HTTP service is offered in form of Restful API, and you can consume it using `curL` or `python.request` or other methods as you prefer.

## Environment Setup

### Prepare Docker Image
Use Dockerfile to build Docker image in your environment. The `Chat`, `Chat Q&A`, and `Summary` use cases share the same Dockerfile. 

All you need to do is to choose the right Dockerfile according to your architecture. The following example is for CPU.

```bash
docker build . -f cpu/Dockerfile -t neuralchat_text_generation:latest
```

If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.


```bash
docker build . -f cpu/Dockerfile -t neuralchat_text_generation:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```  

### Prepare Configuration File and Documents
Before starting NeuralChat services, you need to configure `yaml` file according to your use case. 

- `chat.yaml`: Query Response System
- `chatqna.yaml`: Chat Query & Answer
- `summary.yaml`: Document/Report Summary

In `chat.yaml`, you only need to configure your local ip and host, and set `device` according to your situation.

As for `chatqna.yaml` and `summary.yaml`, you need to prepare a document folder named `rag_docs`, containing the Q&A knowledge files. The rag chatbot will automatically retrieve answers to your question with the context of these files.

Remember to set `device` to `cpu`/`hpu` according to your architecture.


### Start NeuralChat Service
Use the following command to start NeuralChat Text Generation service. 

#### Example of `Chat` Service.

Make sure the specified `port` is available, and `device` is correctly set.

```bash
docker run -it --net=host --ipc=host --name text_gen -v ./chat.yaml:/text_generation.yaml neuralchat_text_generation:latest
```

The specific meaning of each parameter is explaine below:
- `docker run -it`: Create a docker container and launch it interactively.
- `--net=host --ipc=host`: Configure the network of docker container.
- `-v`: Mount `chat.yaml` file the docker container from your local server.
- `--name text_gen`: The name of your docker container, you can set it differently as you need.
- `neuralchat_text_generation:latest`: The name of the Docker image you created just now.

If you need to set proxy settings, add `-e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1"`.

```bash
docker run -it --net=host --ipc=host --name text_gen -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1" -v ./chat.yaml:/text_generation.yaml neuralchat_text_generation:latest
```

#### Example of `Chat Q&A` Service.

```bash
docker run -it --net=host --ipc=host --name text_gen -v ./chatqna.yaml:/text_generation.yaml -v ./rag_docs:/rag_docs neuralchat_text_generation:latest
```

The specific meaning of each parameter is explaine below:
- `-v`: Mount `chatqna.yaml` file into docker container from your local server.
- `-v`: Mount `rag_docs` into the path written in `chatqna.yaml: input_path`

If you need to set proxy settings, add `-e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1"`.

```bash
docker run -it --net=host --ipc=host --name text_gen -v ./chatqna.yaml:/text_generation.yaml -v ./rag_docs:/rag_docs  -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1" neuralchat_text_generation:latest
```


#### Example of `Summary` Service.

```bash
docker run -it --net=host --ipc=host --name text_gen -v ./summary.yaml:/text_generation.yaml -v ./rag_docs:/rag_docs neuralchat_text_generation:latest
```

The specific meaning of each parameter is explaine below:
- `-v`: Mount `summary.yaml` file into docker container from your local server.
- `-v`: Mount `rag_docs` into the path written in `summary.yaml: input_path`

If you need to set proxy settings, add `-e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1"`.

```bash
docker run -it --net=host --ipc=host --name text_gen -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1" -v ./summary.yaml:/text_generation.yaml -v ./rag_docs:/rag_docs neuralchat_text_generation:latest
```


## Consume the Service with Simple Test
when `docker run` command is successfully executed, you can consume the HTTP services offered by NeuralChat. The Restful API of NeuralChat is compatible with OpenAI, so you can use the same request body as OpenAI.


Please substitute `http://127.0.0.1` with your IP and `8000` with the port written in yaml.



### Consume `Chat` Service
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
 -X POST \
 -d '{"model": "Intel/neural-chat-7b-v3-1", "stream": false, "messages": [
  {"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."}]}' \
 -H 'Content-Type: application/json'
```


### Consume `Chat Q&A` Service
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
 -X POST \
 -d '{"model": "Intel/neural-chat-7b-v3-1", "stream": false, "messages": [
  {"role": "user", "content": "What is RAG?"}]}' \
 -H 'Content-Type: application/json'
```

### Consume `Summary` Service
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
 -X POST \
 -d '{"model": "Intel/neural-chat-7b-v3-1", "stream": false, "messages": [
  {"role": "system", "content": "You are a Summary Chatbot designed to help users quickly understand the main points of given texts. Your primary skill is summarization, which involves condensing lengthy information into concise, easily digestible summaries. When users provide you with text, whether it is an article, a document, or any form of written content, your task is to analyze the content and produce a summary that captures the essential information and key points. You should ensure that your summaries are accurate, neutral, and free from personal opinions or interpretations. Your goal is to save users time and make information more accessible by highlighting the most important aspects of the content they are interested in."},
  {"role": "user", "content": "Give me the Summary of Intel 2023 Annual Report."}]}' \
 -H 'Content-Type: application/json'
```
