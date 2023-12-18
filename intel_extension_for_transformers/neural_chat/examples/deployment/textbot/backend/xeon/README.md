This README is intended to guide you through setting up the backend for a text chatbot using the NeuralChat framework. You can deploy this text chatbot on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU and Client GPU.

This textbot shows how to deploy chatbot backend on Intel XEON Scalable Processors.

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

# Configure the textbot.yaml

You can customize the configuration file 'textbot.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item              | Value                                      |
| ------------------- | --------------------------------------- |
| host                | 127.0.0.1                              |
| port                | 8000                                   |
| model_name_or_path  | "Intel/neural-chat-7b-v3-1"        |
| device              | "cpu"                                  |
| tasks_list          | ['textchat']                           |



# Run the TextChat server
To start the TextChat server, use the following command:

```shell
nohup bash run.sh &
```
