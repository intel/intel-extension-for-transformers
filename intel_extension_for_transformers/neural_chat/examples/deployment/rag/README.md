This README is intended to guide you through setting up the server for the AskDoc demo using the NeuralChat framework. You can deploy it on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU and Client GPU.

# Introduction
The popularity of applications like ChatGPT has attracted many users seeking to address everyday problems. However, some users have encountered a challenge known as "model hallucination," where LLMs generate incorrect or nonexistent information, raising concerns about content accuracy. This example introduce our solution to build a retrieval-based chatbot backend server. Though few lines of code, our api could help the user build a local reference database to enhance the accuracy of the generation results.

Before deploying this example, please follow the instructions in the [README](../../README.md) to install the necessary dependencies.

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

You can customize the configuration file 'askdoc.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item                             | Value                                  |
| --------------------------------- | ---------------------------------------|
| host                              | 127.0.0.1                              |
| port                              | 8000                                   |
| model_name_or_path                | "./neural-chat-7b-v3-1"                 |
| device                            | "auto"                                 |
| retrieval.enable                  | true                                   |
| retrieval.args.input_path         | "./docs"                               |
| retrieval.args.persist_dir        | "./example_persist"                    |
| retrieval.args.response_template  | "We cannot find suitable content to answer your query, please contact to find help."    |
| retrieval.args.append             | True        |
| tasks_list                        | ['textchat', 'retrieval']              |


# Run the AskDoc server
The Neural Chat API offers an easy way to create and utilize chatbot models while integrating local documents. Our API simplifies the process of automatically handling and storing local documents in a document store. In this example, we use `./docs/test_doc.txt` for example. You can construct your own retrieval doc of Intel® oneAPI DPC++/C++ Compiler following [this link](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-2/overview.html).


To start the PhotoAI server, run the following command:

```shell
nohup bash run.sh &
```
