Intel Neural Chat Inference Dockerfile installer for Ubuntu22.04

# Start NeuralChat and TGI serving with Docker

## Environment Setup

### Setup Xeon SPR Environment
Use Dockerfile_tgi to build Docker image in your environment.
```bash
docker build . -f Dockerfile_tgi -t neuralchat_tgi:latest
```
If you need to set proxy settings, add `--build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy` like below.
```bash
docker build . -f Dockerfile_tgi -t neuralchat_tgi:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```  

### Start NeuralChat Service
Before starting NeuralChat services, you need to configure `tgi.yaml` according to you read environment.
Make sure the specified `port` is available, `device` is `cpu` (`auto` will not work).
Other detailed parameters please refer to `intel_extension_for_transformers/neural_chat/examples/serving/TGI/README.md`

```bash
docker run -it --net=host --ipc=host -v /var/run/docker.sock:/var/run/docker.sock -v ./tgi.yaml:/tgi.yaml neuralchat_tgi:latest
```


## Consume the Service
when `docker run` command is successfully executed, you can consume the HTTP services offered by NeuralChat.

Here is an example of consuming TGI service, remember to substitute your real ip and port.
```bash
curl ${your_ip}:${your_port}/v1/tgi/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```
