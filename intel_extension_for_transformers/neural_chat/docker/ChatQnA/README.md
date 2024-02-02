# Start NeuralChat and ChatQnA with Docker
Intel Neural Chat ChatQnA Dockerfile installer for Ubuntu22.04/Habana Gaudi/XPU.

Following the instruction of this README.md, you will start a ChatQnA HTTP service with NeuralChat. The whole procedure is very clear and easy for customers to use with only two docker commands.The HTTP service is offered in form of Restful API, and you can consume it using `CurL` or `python.request` or other methods as you prefer.

## Environment Setup

### Prepare Docker Image
Use Dockerfile to build Docker image in your environment.

Remember to choose Dockerfile of your framework (CPU/HPU/XPU), the following example is for CPU.
```bash
cd ./CPU
docker build . -f Dockerfile -t neuralchat_chat_qna:latest
```
If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker build . -f Dockerfile -t neuralchat_chat_qna:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```  

### Prepare Configuration File and Documents
Before starting NeuralChat services, you need to configure `chatqna.yaml` according to you read environment.

As shown in `chatqna.yaml`, you need to prepare a document folder named `rag_docs`, containing the Q&A knowledge files. The rag chatbot will automatically retrieve answers to your question with the context of these files.

Remember to set `device` to `cpu`/`hpu`/`xpu` according to your framework.


### Start NeuralChat Service
Use the following command to start NeuralChat ChatQnA service.

Make sure the specified `port` is available, and `device` is correctly set.

The `-v` command is used to mount `chatqna.yaml` file and your `rag_docs` into the docker container.
```bash
docker run -it --net=host --ipc=host --name chat_qna -v ./chatqna.yaml:/chatqna.yaml -v ./rag_docs:/rag_docs neuralchat_chat_qna:latest
```

If you need to set proxy settings, use the command below.
```bash
docker run -it --net=host --ipc=host --name chat_qna -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY -v ./chatqna.yaml:/chatqna.yaml -v ./rag_docs:/rag_docs neuralchat_chat_qna:latest
```


## Consume the Service
when `docker run` command is successfully executed, you can consume the HTTP services offered by NeuralChat.

Here is an example of consuming ChatQnA service, remember to substitute your real ip and port.
```bash
curl ${your_ip}:${your_port_in_yaml}/v1/askdoc/chat \
  -X POST \
  -d '{"query":"What is Deep Learning?","translated":"What is Deep Learning?","knowledge_base_id":"default","stream":false,"max_new_tokens":128,"return_link":false}' \
  -H 'Content-Type: application/json'
```
