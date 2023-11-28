This README is intended to guide you through setting up the backend for a text chatbot using the NeuralChat framework. You can deploy this text chatbot on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU and Client GPU.

This textbot shows how to deploy chatbot backend on Habana's Gaudi processors (HPU).

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

Install dependencies using pip

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
| model_name_or_path  | "meta-llama/Llama-2-7b-chat-hf"        |
| device              | "hpu"                                  |
| use_deepspeed       | true                                   |
| world_size          | 8                                      |
| tasks_list          | ['textchat']                           |



# Run the TextChat server
To start the TextChat server, use the following command:

```shell
nohup python run_text_chat.py &
```
