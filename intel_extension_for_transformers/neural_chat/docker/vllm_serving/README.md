Intel Neural Chat Inference Dockerfile installer for Ubuntu22.04

# Start NeuralChat and vLLM serving with Docker

## Environment Setup

## Setup NVIDIA GPU environment
Use Dockerfile_vLLM to build Docker image in your environment.
```bash
docker build . -f Dockerfile_vllm -t neuralchat_vllm:latest
```
If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker build . -f Dockerfile_vllm -t neuralchat_vllm:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```

### Start NeuralChat Service
Before starting NeuralChat services, you need to configure `vllm.yaml` according to you read environment.
Make sure the specified `port` is available, `device` is `cuda` (`auto` will not work).
```bash
docker run -it --runtime=nvidia --gpus all --net=host --ipc=host -v /var/run/docker.sock:/var/run/docker.sock -v ./vllm.yaml:/vllm.yaml neuralchat_vllm:latest
```
If you need to set proxy settings, add `-e http_proxy=<your proxy> -e https_proxy=<your proxy>` like below.
```bash
docker run -it --runtime=nvidia --gpus all -e http_proxy=<your proxy> -e https_proxy=<your proxy> --net=host --ipc=host -v /var/run/docker.sock:/var/run/docker.sock -v ./vllm.yaml:/vllm.yaml neuralchat_vllm:latest
```

## Consume the Service
when `docker run` command is successfully executed, you can consume the HTTP services offered by NeuralChat.

Here is an example of consuming vLLM service, remember to substitute your real ip and port.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Tell me about Intel Xeon processors."}' http://localhost:8000/v1/chat/completions
```
