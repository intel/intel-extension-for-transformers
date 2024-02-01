# Introduction

Intel Extension for Transformers provides a comprehensive suite of Langchain-based extension APIs, including advanced retrievers, embedding models, and vector stores. These enhancements are carefully crafted to expand the capabilities of the original langchain API, ultimately boosting overall performance. This extension is specifically tailored to enhance the functionality and performance of RAG.


We have introduced enhanced vector store operations, allowing users to adjust and fine-tune their settings even after the chatbot has been initialized, providing a more adaptable and user-friendly experience. For Langchain users, integrating and utilizing optimized Vector Stores is straightforward by replacing the original Chroma API in Langchain.

We offer optimized retrievers such as `VectorStoreRetriever` and `ChildParentRetriever` to efficiently handle vector store operations, ensuring optimal retrieval performance. Additionally, we provide quantized embedding models to accelerate embedding documents. These Langchain extension APIs are easy to use and are optimized for both performance and accuracy, specifically tailored for Intel hardware.

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

Install retrieval plugin dependencies using pip:
```bash
pip install -r ../../pipeling/plugins/retrieval/requirements.txt
```

# Test

```shell
python main.py
```
