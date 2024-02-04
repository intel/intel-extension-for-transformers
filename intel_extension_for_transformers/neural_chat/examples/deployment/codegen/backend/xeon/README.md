This README serves as a guide to set up the backend for a code generation chatbot utilizing the NeuralChat framework. You can deploy this code generation chatbot across various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU, and Client GPU.

This code generation chatbot demonstrates how to deploy it specifically on Intel XEON processors. To run the 34b or 70b level LLM model, we require implementing model parallelism using multi-node strategy.


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

# Install ITREX

```shell
pip install intel-extension-for-transformers
```

# Install Python Dependencies

Install neuralchat dependencies:

```bash
pip install -r ../../../requirements_cpu.txt
```

# Configure the codegen.yaml

You can customize the configuration file 'codegen.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item              | Value                                      |
| ------------------- | --------------------------------------- |
| host                | 0.0.0.0                              |
| port                | 8000                                   |
| model_name_or_path  | "ise-uiuc/Magicoder-S-DS-6.7B"        |
| device              | "cpu"                                  |
| tasks_list          | ['codegen']                           |


# Run the Code Generation Chatbot Server

To start the code-generating chatbot server, use the following command:

```shell
nohup bash run.sh &
```
