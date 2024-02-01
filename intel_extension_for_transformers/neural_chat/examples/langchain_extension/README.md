# Introduction

Intel Extension for Transformers provides a comprehensive suite of Langchain-based extension APIs, including advanced retrievers, embedding models, and vector stores. These enhancements are carefully crafted to expand the capabilities of the original langchain API, ultimately boosting overall performance. This extension is specifically tailored to enhance the functionality and performance of RAG.

We introduce enhanced vector store operations, enabling users to adjust and fine-tune their settings even after the chatbot has been initialized, offering a more adaptable and user-friendly experience. For langchain users, integrating and utilizing optimized Vector Stores is straightforward by replacing the original Chroma API in langchain.
We provide optimized retrievers such as `VectorStoreRetriever`, `ChildParentRetriever` to efficiently handle vectorstore operations, ensuring optimal retrieval performance.
We also provide quantized embedding models to speed up the retrieval inference. These langchain extension APIs are easy to use and optimized for performance or accuracy on Intel hardware.

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
