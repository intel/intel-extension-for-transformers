
This README is intended to guide you through setting up a NeuralChat chatbot with TGI serving framework. You can deploy it on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU and Client GPU.

# Introduction
Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and T5.

NeuralChat is now able to integrate TGI serving and offer customers TGI-format Restful APIs like /v1/generate and /v1/generate_stream. Going through this example, you could start a NeuralChat service with the plugins and frameworks you configure in tgi.yaml.


# Setup Environment

## Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
```

## Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

## Install Python dependencies

Install the following Python dependencies using Conda:

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
conda install git-lfs -y
```

Install other dependencies using pip:

```bash
pip install -r ../../../requirements.txt
```


## Download Models
```shell
git-lfs install
git clone https://huggingface.co/Intel/neural-chat-7b-v3-1
```


# Configure YAML

You can customize the configuration file 'tgi.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item                             | Value                                  |
| --------------------------------- | ---------------------------------------|
| host                              | 0.0.0.0                              |
| port                              | 8000                                   |
| model_name_or_path                | "./neural-chat-7b-v3-1"                 |
| device                            | "cpu"/"gpu"/"hpu"                                 |
| serving.framework                  | "tgi"                                   |
| serving.framework.tgi_engine_params.endpoint        | Your existed tgi service endpoint. when endpoint is set, neuralchat will not start a tgi service, and other params will not work any more.                |
| serving.framework.tgi_engine_params.port        | 9876, the port that neuralchat will help to start tgi service.                    |
| serving.framework.tgi_engine_params.sharded        | true (false only on cpu)                    |
| serving.framework.tgi_engine_params.num_shard  | 4 (not effective when sharded is false)    |
| serving.framework.tgi_engine_params.habana_visible_devices      | "0,1" (only on hpu)        |


# Run the NeuralChat server with TGI framework

To start the NeuralChat server with TGI framework, run the following command:

```shell
nohup bash run.sh &
```


# Consume the Services
After the services are successfully launched, you can consume the HTTP services offered by NeuralChat.

Here is an example of consuming TGI service, remember to substitute your real ip and port.

```bash
curl ${your_ip}:${your_port}/v1/tgi/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```

Of course, you can also consume the service via `postman`, `http request`, or other ways.

If neuralchat is unable to call your local tgi service, try the command below then try again.
```bash
unset http_proxy
```
