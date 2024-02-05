Intel Neural Chat Code Generation Dockerfile installer for Ubuntu22.04/Habana Gaudi/XPU.

# Start NeuralChat and Code Generation Service with Docker

## Environment Setup

### Setup Xeon SPR Environment
Use Dockerfile to build Docker image in your environment.

Remember to choose Dockerfile of your framework (CPU/HPU/XPU), the following example is for CPU.
```bash
cd ./cpu
docker build . -f Dockerfile -t neuralchat_codegen:latest
```
If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker build . -f Dockerfile -t neuralchat_codegen:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```  

### Prepare Configuration File and Documents
Before starting NeuralChat services, you need to configure `codegen.yaml` according to you read environment.


Specify your available host ip and port, and a code-generation model (`ise-uiuc/Magicoder-S-DS-6.7B` is recommended for its accuracy). Remember to set `device` to `cpu`/`hpu`/`xpu` according to your framework.


### Start NeuralChat Service
Use the following command to start NeuralChat CodeGen service.

Make sure the specified `port` is available, and `device` is correctly set.
```bash
docker run -it --net=host --ipc=host --name code_gen -v ./codegen.yaml:/codegen.yaml neuralchat_codegen:latest
```


## Consume the Service
when `docker run` command is successfully executed, you can consume the HTTP services offered by NeuralChat.

Here is an example of consuming CodeGen service, remember to substitute your real ip and port.
```bash
curl ${your_ip}:${your_port_in_yaml}/v1/code_generation \
  -X POST \
  -d '{"prompt":"def print_hello_world():"}' \
  -H 'Content-Type: application/json'
```
