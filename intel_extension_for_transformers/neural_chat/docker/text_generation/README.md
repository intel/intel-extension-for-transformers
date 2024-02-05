# Start NeuralChat Text Generation Service with Docker
Intel Neural Chat Text Generation Dockerfile installer for Ubuntu22.04/Habana Gaudi.

Following the instruction of this README.md, you will start a Text Generation HTTP service with NeuralChat. The whole procedure is very clear and easy for customers to use with only two docker commands.The HTTP service is offered in form of Restful API, and you can consume it using `CurL` or `python.request` or other methods as you prefer.

## Environment Setup

### Prepare Docker Image
Use Dockerfile to build Docker image in your environment. The `chat`, `chat q&a`, and `document/report summary` use cases share the same Dockerfile. 

All you need to do is to choose the right Dockerfile according to your architecture. The following example is for CPU.
```bash
cd ./cpu
docker build . -f Dockerfile -t neuralchat_text_generation:latest
```
If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker build . -f Dockerfile -t neuralchat_text_generation:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
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
Use the following command to start NeuralChat Text Generation service. The example of starting a `chatqna` service is represented as below.

Make sure the specified `port` is available, and `device` is correctly set.

```bash
docker run -it --net=host --ipc=host --name chat_qna -v ./chatqna.yaml:/text_generation.yaml -v ./rag_docs:/rag_docs neuralchat_chat_qna:latest
```
The specific meaning of each parameter is explaine below:
- `docker run -it`: Create a docker container and launch it interactively.
- `--net=host --ipc=host`: Configure the network of docker container.
- `-v`: Mount `chatqna.yaml` file and your `rag_docs` into the docker container from your local server.
- `--name chat_qna`: The name of your docker container, you can set it differently as you need.
- `neuralchat_chat_qna:latest`: The name of the Docker image you created just now.

If you need to set proxy settings, use the command below.
```bash
docker run -it --net=host --ipc=host --name chat_qna -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY -v ./chatqna.yaml:/text_generation.yaml -v ./rag_docs:/rag_docs neuralchat_chat_qna:latest
```


## Consume the Service
when `docker run` command is successfully executed, you can consume the HTTP services offered by NeuralChat. The Restful API of NeuralChat is compatible with OpenAI, so you can use the same request body as OpenAI.

### Consume Chat Service
```bash
curl ${your_ip}:${your_port_in_yaml}/v1/chat/completions \
 -X POST \
 -d '{"model": "Intel/neural-chat-7b-v3-1", "stream": true, "messages": [
  {"role": "user", "content": "Tell me about Intel Xeon Scalable Processors."}]}' \
 -H 'Content-Type: application/json'
```


### Consume ChatQnA Service
```bash
curl ${your_ip}:${your_port_in_yaml}/v1/chat/completions \
 -X POST \
 -d '{"model": "Intel/neural-chat-7b-v3-1", "stream": true, "messages": [
  {"role": "user", "content": "What is RAG?"}]}' \
 -H 'Content-Type: application/json'
```

### Consume Summary Service
```bash
curl ${your_ip}:${your_port_in_yaml}/v1/chat/completions \
 -X POST \
 -d '{"model": "Intel/neural-chat-7b-v3-1", "stream": true, "messages": [
  {"role": "system", "content": "You are a Summary Chatbot designed to help users quickly understand the main points of given texts. Your primary skill is summarization, which involves condensing lengthy information into concise, easily digestible summaries. When users provide you with text, whether it is an article, a document, or any form of written content, your task is to analyze the content and produce a summary that captures the essential information and key points. You should ensure that your summaries are accurate, neutral, and free from personal opinions or interpretations. Your goal is to save users time and make information more accessible by highlighting the most important aspects of the content they are interested in."},
  {"role": "user", "content": "Give me the Summary of Intel 2023 Annual Report."}]}' \
 -H 'Content-Type: application/json'
```
