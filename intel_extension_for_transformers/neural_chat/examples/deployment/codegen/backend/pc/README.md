This README is designed to walk you through setting up the backend for a code-generating chatbot using the NeuralChat framework. You can deploy this chatbot on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU, and Client GPU.

This code-generating chatbot demonstrates how to deploy it specifically on a Laptop PC. To ensure smooth operation on a laptop, we need to implement [LLM runtime optimization](../../../../../../llm/runtime/graph/README.md) to accelerate the inference process.

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
pip install ../../../../../requirements_pc.txt
pip install transformers==4.34.1
```

# Configure the codegen.yaml

You can customize the configuration file 'codegen.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

| Item               | Value                                |
| ------------------ | -------------------------------------|
| host               | 127.0.0.1                            |
| port               | 8000                                 |
| model_name_or_path | "codellama/CodeLlama-7b-hf"          |
| device             | "cpu"                                |
| tasks_list         | ['textchat']                         |
| optimization       |                                      |
|                    |  use_llm_runtime  | true             |
|                    |  optimization_type| "weight_only"    |
|                    |  compute_dtype    | "int8"           |
|                    |  weight_dtype     | "int4"           |



# Run the Code Generation Chatbot server
To start the code-generating chatbot server, use the following command:

```shell
nohup python run_code_gen.py &
```
