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

ARG ITREX_VER=main
ARG PYTHON_VERSION=3.9
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git
ARG REPO_PATH=""

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install system dependencies
SHELL ["/bin/bash", "--login", "-c"]
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget numactl git \
    && apt-get install -y openssh-server \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download ITREX code
RUN mkdir -p /intel-extension-for-transformers
COPY ${REPO_PATH} /intel-extension-for-transformers
RUN if [ "$REPO_PATH" == "" ]; then rm -rf intel-extension-for-transformers/* && rm -rf intel-extension-for-transformers/.* ; git clone --single-branch --branch=${ITREX_VER} ${REPO} intel-extension-for-transformers ; fi
WORKDIR /intel-extension-for-transformers

RUN source activate && conda activate neuralchat && pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable-cpu && \
    cd /intel-extension-for-transformers && pip install -r requirements.txt && pip install -v . && \
    cd ./intel_extension_for_transformers/neural_chat/examples/finetuning/instruction && pip install -r requirements.txt && \
    cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat && pip install -r requirements_cpu.txt && \
    pip install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi future six requests dataclasses && \
    pip install jemalloc gperftools && \
    pip install typing_extensions datasets accelerate SentencePiece evaluate nltk rouge_score protobuf==3.20.1 tokenizers einops peft

# Enable passwordless ssh for mpirun
RUN mkdir /var/run/sshd
RUN passwd -d root
RUN sed -i'' -e's/^#PermitRootLogin prohibit-password$/PermitRootLogin yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PasswordAuthentication yes$/PasswordAuthentication yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PermitEmptyPasswords no$/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config \
        && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
EXPOSE 22

WORKDIR /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/

CMD ["/usr/sbin/sshd", "-D"]

# HABANA environment
FROM vault.habana.ai/gaudi-docker/1.12.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest as hpu

ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/root:/usr/lib/habanalabs/
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git
ARG REPO_PATH=""
ARG ITREX_VER=main

RUN apt-get update && \
    apt-get install git-lfs && \
    git-lfs install

# Download ITREX code
SHELL ["/bin/bash", "--login", "-c"]
RUN mkdir -p /intel-extension-for-transformers
COPY ${REPO_PATH} /intel-extension-for-transformers
RUN if [ "$REPO_PATH" == "" ]; then rm -rf /intel-extension-for-transformers/* && rm -rf /intel-extension-for-transformers/.* ; git clone --single-branch --branch=${ITREX_VER} ${REPO} /intel-extension-for-transformers ; fi
WORKDIR /intel-extension-for-transformers

# Build ITREX
RUN cd /intel-extension-for-transformers && \
    sed -i '/find-links https:\/\/download.pytorch.org\/whl\/torch_stable.html/d' requirements.txt && \
    sed -i '/^torch/d;/^intel-extension-for-pytorch/d' requirements.txt && \
    pip install -r requirements.txt && \
    pip install -v .

COPY intel_extension_for_transformers/neural_chat/docker/inference/0001-fix-lm-head-overriden-issue-move-it-from-checkpoint-.patch /

RUN git clone https://github.com/huggingface/optimum-habana.git && \
    cd optimum-habana/ && git reset --hard 0fb6bb6b7bbfb06b85a6c10f113b8283aa028b02 && pip install -e . && cd ../ && \
    cd ./optimum-habana/examples/text-generation/ && \
    pip install -r requirements.txt && \
    cd / && \
    pip install peft && \
    pip install einops && \
    pip install datasets && pip list && \
    git clone https://github.com/HabanaAI/DeepSpeed.git && \
    cd DeepSpeed && \
    git checkout -b v1.12 origin/1.12.0 && \
    git apply /0001-fix-lm-head-overriden-issue-move-it-from-checkpoint-.patch && \ 
    pip install -e .

# Install dependency
RUN cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/examples/finetuning/instruction && \
    sed -i '/find-links https:\/\/download.pytorch.org\/whl\/torch_stable.html/d' requirements.txt && \
    sed -i '/^torch/d;/^intel-extension-for-pytorch/d' requirements.txt && \
    pip install -r requirements.txt && \
    cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/ && \
    pip install -r requirements_hpu.txt && \
    pip install transformers==4.34.1 && \
    pip install accelerate==0.23.0 && \
    pip install pymysql

WORKDIR /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/

## NVIDIA GPU environment
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 as nvgpu

ARG ITREX_VER=main
ARG PYTHON_VERSION=3.9
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install system dependencies
SHELL ["/bin/bash", "--login", "-c"]
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget numactl git \
    && apt-get install -y openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download ITREX code
RUN git clone --single-branch --branch=${ITREX_VER} ${REPO} intel-extension-for-transformers

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
    conda create -yn neuralchat python=${PYTHON_VERSION} && \
    echo "conda activate neuralchat" >> ~/.bashrc && \
    echo "export PYTHONPATH=/intel-extension-for-transformers" >> ~/.bashrc && \
    source ~/.bashrc

RUN source activate && conda activate neuralchat && cd /intel-extension-for-transformers && \
    pip install -r ./intel_extension_for_transformers/neural_chat/examples/finetuning/instruction/requirements.txt && \
    pip install -r ./intel_extension_for_transformers/neural_chat/requirements.txt

WORKDIR /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/

CMD ["/usr/sbin/sshd", "-D"]