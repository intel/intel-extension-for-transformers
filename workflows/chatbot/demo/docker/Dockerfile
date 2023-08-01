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
#   docker build ./ -f Dockerfile -t chatbotdemo:latest
# If you need to use proxy, please use the following command
#   docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f Dockerfile -t chatbotdemo:latest

ARG UBUNTU_VER=22.04
FROM ubuntu:${UBUNTU_VER} as demo

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ARG ITREX_VER=main
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    python3 \
    python3-pip \
    python3-dev \
    python3-distutils \
    autoconf \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    numactl \
    time \
    wget \
    vim

# Install Java environment for elasticsearch
RUN apt-get install default-jre -y
RUN apt-get install default-jdk -y

# Install and upgrade node and npm for frontend
RUN apt-get install npm -y && \
    npm install -g n && \
    n stable && \
    hash -r && \
    npm install -g npm@latest

RUN ln -sf $(which python3) /usr/bin/python

RUN python -m pip --no-cache-dir install --upgrade pip
RUN python -m pip install --no-cache-dir setuptools

# Download ITREX code
RUN git clone --single-branch --branch=${ITREX_VER} ${REPO} itrex

# Install miniconda
ENV CONDA_DIR /root/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /root/conda

# Put conda in path so we can use conda activate
SHELL ["/bin/bash", "--login", "-c"]
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash && \
    unset -f conda && \
    export PATH=$CONDA_DIR/bin/:${PATH} && \
    conda config --add channels intel && \
    conda create -yn chatbot-demo python=3.9 && \
    echo "conda activate chatbot-demo" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c", "conda", "run", "-n", "chatbot-demo"]

RUN conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y && \
    conda install jemalloc gperftools -c conda-forge -y && \
    conda install pytorch torchvision torchaudio cpuonly -c pytorch && \
    pip install farm-haystack==1.14.0 && \
    pip install intel_extension_for_pytorch && \
    pip install transformers diffusers accelerate SentencePiece peft evaluate nltk datasets && \
    pip install fastapi uvicorn sse_starlette bottle gevent pymysql && \
    pip install schema && \
    pip install -i https://test.pypi.org/simple/ intel-extension-for-transformers==1.0.0.dev20230602


# Add user
SHELL ["/bin/bash", "--login", "-c"]
RUN adduser --disabled-password --gecos "" fastrag

# Switch to non-root user
USER fastrag

# Change directory
WORKDIR /home/fastrag/

# Download Elasticsearch
ADD https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-linux-x86_64.tar.gz .

# Switch to root user
USER root

WORKDIR /itrex/workflows/chatbot/demo/
