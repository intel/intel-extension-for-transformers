This example demonstrates how to use natural language to generate SQL in NeuralChat. In this example, we use the SQLCoder for this SQL generation task.

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

## Install Intel Extension for Transformers

```shell
pip install intel-extension-for-transformers
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
pip install -r ../../requirements.txt
```

# Test

```shell
python main.py
```
