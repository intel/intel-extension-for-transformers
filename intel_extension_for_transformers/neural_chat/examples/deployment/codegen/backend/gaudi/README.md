This README serves as a guide to set up the backend for a code generation chatbot utilizing the NeuralChat framework. You can deploy this code generation chatbot across various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU, and Client GPU.

This code generation chatbot demonstrates how to deploy it specifically on Habana's Gaudi processors (HPU). To run the 34b or 70b level LLM model, we require implementing model parallelism using Gaudi multi-cards.

# Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
```

# Install Python dependencies

Install dependencies using pip

>**Note**: Please make sure transformers version is 4.34.1
```bash
pip install ../../../../../requirements_hpu.txt
pip install transformers==4.34.1
```

# Configure the codegen.yaml

You can customize the configuration file 'codegen.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item              | Value                                      |
| ------------------- | --------------------------------------- |
| host                | 127.0.0.1                              |
| port                | 8000                                   |
| model_name_or_path  | "Phind/Phind-CodeLlama-34B-v2"        |
| device              | "hpu"                                  |
| use_deepspeed       | true                                   |
| world_size          | 8                                      |
| tasks_list          | ['textchat']                           |



# Run the Code Generation Chatbot server
To start the code-generating chatbot server, use the following command:

```shell
nohup python run_code_gen.py &
```
