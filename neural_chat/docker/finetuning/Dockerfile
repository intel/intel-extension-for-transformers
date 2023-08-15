# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.
#
# ============================================================================
# How to build: 
#   docker build ./ -f Dockerfile -t chatbot_finetune:latest
# If you need to use proxy, please use the following command
#   docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f Dockerfile -t chatbot_finetune:latest

## SPR environment
ARG UBUNTU_VER=22.04
FROM ubuntu:${UBUNTU_VER} as cpu

ARG ITREX_VER=neural_chat_api
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install system dependencies
SHELL ["/bin/bash", "--login", "-c"]
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget numactl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download ITREX code
RUN git clone --single-branch --branch=${ITREX_VER} ${REPO} itrex

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
SHELL ["/bin/bash", "--login", "-c"]
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash && \
    unset -f conda && \
    export PATH=$CONDA_DIR/bin/:${PATH} && \
    conda config --add channels intel && \
    conda create -yn chatbot-finetuning python=3.9 && \
    echo "conda activate chatbot-finetuning" >> ~/.bashrc && \
    source ~/.bashrc

RUN source activate && conda activate chatbot-finetuning && pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable-cpu && \
    pip install datasets torch accelerate SentencePiece evaluate nltk rouge_score protobuf==3.20.1 tokenizers einops && \
    git clone https://github.com/huggingface/peft.git && cd peft && python setup.py install && \
    cd /itrex && pip install -v . && \
    cd neural_chat/docker/finetuning/ && pip install -r requirements.txt

WORKDIR /itrex/neural_chat/docker/finetuning/


# HABANA environment
FROM vault.habana.ai/gaudi-docker/1.10.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest as hpu

ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/root:/usr/lib/habanalabs/

# Install optimum-habana
RUN git clone https://github.com/huggingface/optimum-habana.git && \
    cd ./optimum-habana/examples/text-generation/ && \
    pip install -r requirements.txt && \
    apt-get update && \
    apt-get install git-lfs && \
    git-lfs install

RUN pip install git+https://github.com/huggingface/optimum-habana.git && \
    pip install einops && \
    pip install datasets && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.10.0 && \
    git clone https://github.com/huggingface/peft.git && cd peft && python setup.py install

# Download ITREX code
ARG ITREX_VER=neural_chat_api
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git

RUN git clone --single-branch --branch=${ITREX_VER} ${REPO} itrex && \
    cd /itrex/neural_chat/docker/finetuning/ && \
    pip install -r requirements.txt
    
# Build ITREX
RUN cd /itrex && pip install -v . && \
    pip install transformers==4.28.1

WORKDIR /itrex/neural_chat/docker/finetuning/
