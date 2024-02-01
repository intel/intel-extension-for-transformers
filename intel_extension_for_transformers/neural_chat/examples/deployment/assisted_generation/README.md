This README serves as a guide to set up the backend for an assisted generation chatbot based on the assisted_model of transformers utilizing the NeuralChat framework. You can deploy this assisted generation chatbot across various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU, and Client GPU.

This assisted generation chatbot example demonstrates how to deploy an assisted chatbot. Make sure your LLM model and the assisted model have the same model architecture.


# Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash `Miniconda*.sh`
source ~/.bashrc
# Create Conda Virtual Environment
conda create -n your_env_name python=3.9 -y
conda activate your_env_name
```

# Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

# Install Python Dependencies

Install Conda dependencies

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
```

Install neuralchat dependencies

```bash
pip install -r ../../../requirements.txt
pip install transformers==4.35.2
```


# Configure the assisted_gen.yaml

You can customize the configuration file 'codegen.yaml' to match your environment setup. Here's a table to help you understand the configurable options.
Make sure the assistant_model has the same architecture of your model. (Such as the opt-arch or the Llama-arch)

|  Item              | Value                                      |
| ------------------- | --------------------------------------- |
| host                | 0.0.0.0                              |
| port                | 8000                                   |
| model_name_or_path  | "facebook/opt-13b"           |
| device              | "cpu"                                  |
| assistant_model       | "facebook/opt-350m"                        |
| tasks_list          | ['textchat']                           |


# Run the Code Generation Chatbot Server

To start the assisted chatbot server, use the following command:

```shell
nohup bash run.sh &
```
