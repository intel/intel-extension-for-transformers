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
#   docker build ./ -f Dockerfile -t document_indexing:latest
# If you need to use proxy, please use the following command
#   docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f Dockerfile -t document_indexing:latest

ARG UBUNTU_VER=22.04
FROM ubuntu:${UBUNTU_VER} as chatbot

ARG ITREX_VER=main
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install system dependencies
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
    conda create -yn document_indexing python=3.9 && \
    echo "conda activate document_indexing" >> ~/.bashrc && \
    source ~/.bashrc

SHELL ["/bin/bash", "--login", "-c", "conda", "run", "-n", "chatbot-demo"]
RUN pip install langchain chromadb PyPDF2 farm-haystack python-docx pandas InstructorEmbedding

WORKDIR /itrex/workflows/chatbot/inference/document_indexing/
