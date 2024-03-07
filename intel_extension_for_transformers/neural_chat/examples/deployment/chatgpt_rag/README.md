This example guides you through setting up the backend for a text chatbot using the NeuralChat framework and OpenAI LLM models such as `gpt-3.5-turbo` or `gpt-4`.
Also, this example shows you how to feed your own corpus to RAG (Retrieval Augmented Generation) with NeuralChat retrieval plugin.

# Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
```

# Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

# Install Python dependencies

Install the following Python dependencies using Conda:

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
```

Install other dependencies using pip

>**Note**: Please make sure transformers version is 4.34.1
```bash
pip install -r ../../../requirements.txt
pip install transformers==4.34.1
```

# Configure YAML

You can customize the configuration file 'askdoc.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item                             | Value                                  |
| --------------------------------- | ---------------------------------------|
| host                              | 127.0.0.1                              |
| port                              | 8000                                   |
| model_name_or_path                | "gpt-3.5-turbo"                 |
| device                            | "auto"                                 |
| retrieval.enable                  | true                                   |
| retrieval.args.input_path         | "./docs"                               |
| retrieval.args.persist_dir        | "./example_persist"                    |
| retrieval.args.response_template  | "We cannot find suitable content to answer your query, please contact to find help."    |
| retrieval.args.append             | True        |
| tasks_list                        | ['textchat', 'retrieval']              |


# Configure OpenAI keys

Set your `OPENAI_API_KEY` and `OPENAI_ORG` environment variables (if applicable) for using OpenAI models.

```
export OPENAI_API_KEY=xxx
export OPENAI_ORG=xxx
```

# Run the TextChat server
To start the TextChat server, use the following command:

```shell
nohup bash run.sh &
```

# Test the TextChat server

curl http://localhost:8000/v1/chat/completions     -H "Content-Type: application/json"     -d '{
    "model": "Intel/neural-chat-7b-v3-1",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What are the key features of the Intel® oneAPI DPC++/C++ Compiler?"
        }
    ],
    "max_tokens": 20
}'
