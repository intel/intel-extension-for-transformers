This README serves as a guide to set up the backend for a code generation chatbot utilizing the NeuralChat framework. You can deploy this code generation chatbot across various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU, and Client GPU.

This code generation chatbot demonstrates how to deploy it specifically on Intel XEON processors using [intel exteion for pytorch](https://github.com/intel/intel-extension-for-pytorch) BFloat16 optimization. 

# Setup Conda

First, you need to install and configure the Conda environment:

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
conda create -n demo python=3.9
conda activate demo
```

# Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

# Install ITREX

```bash
git clone https://github.com/intel/intel-extension-for-transformers.git
cd ./intel-extension-for-transformers/
python setup.py install
```

# Install NeuralChat Python Dependencies

Install neuralchat dependencies:

```bash
pip install -r ../../../../../../requirements_cpu.txt
```

# Install Python dependencies
```bash
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
```

# Configure the codegen.yaml

You can customize the configuration file 'codegen.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item               | Value                                      |
| ------------------- | ------------------------------------------ |
| host                | 0.0.0.0                                    |
| port                | 8000                                       |
| model_name_or_path  | "ise-uiuc/Magicoder-S-DS-6.7B"             |
| device              | "cpu"                                      |
| tasks_list          | ['codegen']                                |

Note: To switch from code generation to text generation mode, adjust the model_name_or_path settings accordingly, e.g. the model_name_or_path can be set "Intel/neural-chat-7b-v3-3".


# Run the Code Generation Chatbot Server

To start the code-generating chatbot server, use the following command:

```shell
bash run.sh
```

Note: Please adapt the core number in the commands `export OMP_NUM_THREADS=48` and `numactl -l -C 0-47 python -m run_code_gen` based on your CPU specifications for the `run.sh` script, which can be checked using `lscpu`.
