Intel Neural Chat Code Generation Dockerfile installer for Ubuntu22.04/Habana Gaudi.

# Start NeuralChat and Code Generation Service with Docker

## Environment Setup

### Setup Xeon SPR Environment
Use Dockerfile to build Docker image in your environment.

Remember to choose Dockerfile of your framework (CPU/HPU), the following example is for CPU.
```bash
docker build . -f cpu/Dockerfile -t neuralchat_codegen:latest
```

If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker build . -f cpu/Dockerfile -t neuralchat_codegen:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```  

### Prepare Configuration File and Documents
Before starting NeuralChat services, you need to configure `codegen.yaml` according to you read environment.


Specify your available host ip and port, and a code-generation model (`ise-uiuc/Magicoder-S-DS-6.7B` is recommended for its accuracy). Remember to set `device` to `cpu`/`hpu` according to your framework.


### Start NeuralChat Service
Use the following command to start NeuralChat CodeGen service.


```bash
docker run -it --net=host --ipc=host --name code_gen -v ./codegen.yaml:/codegen.yaml neuralchat_codegen:latest
```

Make sure the specified `port` is available, and `device` is correctly set.


If you need to set proxy settings, add `-e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1"`.

```bash
docker run -it --net=host --ipc=host --name code_gen -v ./codegen.yaml:/codegen.yaml -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1" neuralchat_codegen:latest
```


## Consume the Service with Simple Test
After `docker run` command is successfully executed, you can consume the HTTP services offered by NeuralChat. You can start a new terminal to enter the docker container for further jobs.

Here is an example of consuming CodeGen service, remember to substitute `http://127.0.0.1` with your IP and `8000` with the port written in yaml.

```bash
# start a new terminal to enter the container
docker exec -it code_gen /bin/bash

curl http://127.0.0.1:8000/v1/code_generation \
  -X POST \
  -d '{"prompt":"def print_hello_world():"}' \
  -H 'Content-Type: application/json'
```
