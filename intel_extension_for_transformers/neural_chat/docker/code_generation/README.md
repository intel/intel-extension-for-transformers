Intel Neural Chat Code Generation Dockerfile installer for Ubuntu22.04/Habana Gaudi.

# Start NeuralChat and Code Generation Service with Docker

## Environment Setup

### Setup Xeon SPR Environment
Use Dockerfile to build Docker image in your environment.

Remember to choose Dockerfile of your framework (CPU/HPU), the following example is for CPU.

If your environment requires a proxy to access the internet, export your development system's proxy settings to the docker environment:

```bash
export DOCKER_BUILD_ARGS="--build-arg https_proxy=$https_proxy \
       --build-arg http_proxy=$http_proxy \
       --build-arg no_proxy=$no_proxy"

docker build . -f cpu/Dockerfile \
       ${DOCKER_BUILD_ARGS} \
      -t intel/intel-extension-for-transformers:code-generation-cpu-1.4.0
```  
Or pull the docker image as follows:

```bash
docker pull intel/intel-extension-for-transformers:code-generation-cpu-1.4.0
```

### Prepare Configuration File and Documents
Before starting NeuralChat services, you need to configure `codegen.yaml` according to you read environment.


Specify your available host ip and port, and a code-generation model (`ise-uiuc/Magicoder-S-DS-6.7B` is recommended for its accuracy). Remember to set `device` to `cpu`/`hpu` according to your framework.


### Start NeuralChat Service
Use the following command to start NeuralChat CodeGen service.

If your environment requires a proxy to access the internet, export your development system's proxy settings to the docker environment:

```bash
export DOCKER_RUN_ENVS="-e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1"

docker run -it --net=host \
      --ipc=host \
      --name code_gen -v ./codegen.yaml:/codegen.yaml \
      ${DOCKER_RUN_ENVS} \
      intel/intel-extension-for-transformers:code-generation-cpu-1.4.0
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
