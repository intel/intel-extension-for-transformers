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
ARG PYTHON_VERSION=3.10
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git
ARG REPO_PATH=""
ARG SSHD_PORT=22
ENV SSHD_PORT ${SSHD_PORT}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install system dependencies
SHELL ["/bin/bash", "--login", "-c"]
RUN apt update \
    && apt install -y build-essential wget numactl git openssh-server libgl1-mesa-glx libjemalloc2 google-perftools \
    && apt install -y python${PYTHON_VERSION} python3-pip \
    && pip install --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Download ITREX code
RUN mkdir -p /intel-extension-for-transformers
COPY ${REPO_PATH} /intel-extension-for-transformers
RUN if [ "$REPO_PATH" == "" ]; then rm -rf intel-extension-for-transformers/* && rm -rf intel-extension-for-transformers/.* ; git clone --single-branch --branch=${ITREX_VER} ${REPO} intel-extension-for-transformers ; fi
WORKDIR /intel-extension-for-transformers

RUN pip install oneccl_bind_pt --extra-index-url  https://pytorch-extension.intel.com/release-whl/stable/cpu/us/ && \
    cd /intel-extension-for-transformers && pip install schema==0.7.5 numpy==1.26.4 && \
    python setup.py install && \
    cd ./intel_extension_for_transformers/neural_chat/examples/finetuning/instruction && pip install -r requirements.txt && \
    cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat && pip install -r requirements_cpu.txt && \
    pip install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi future six requests dataclasses && \
    pip install typing_extensions datasets accelerate SentencePiece evaluate nltk rouge_score protobuf==3.20.1 tokenizers einops peft

# Enable passwordless ssh for mpirun
RUN mkdir /var/run/sshd
RUN passwd -d root
RUN sed -i'' -e's/^#PermitRootLogin prohibit-password$/PermitRootLogin yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PasswordAuthentication yes$/PasswordAuthentication yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PermitEmptyPasswords no$/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config \
        && echo "Port "$SSHD_PORT"" >> /etc/ssh/sshd_config \
        && echo "Host *" >> /etc/ssh/ssh_config \
        && echo "  Port "$SSHD_PORT"" >> /etc/ssh/ssh_config \
        && echo "  StrictHostKeyChecking no" >> /etc/ssh/ssh_config
EXPOSE ${SSHD_PORT}

WORKDIR /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/

CMD ["/usr/sbin/sshd", "-D"]

# HABANA environment
FROM vault.habana.ai/gaudi-docker/1.13.0/ubuntu22.04/habanalabs/pytorch-installer-2.1.0:latest as hpu

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
    pip install schema==0.7.5 numpy==1.26.4 && \
    python setup.py install

RUN git clone https://github.com/huggingface/optimum-habana.git && \
    cd optimum-habana/ && git reset --hard 0dfda27dd0351d0753c35de2abb752309e2383b4 && pip install -e . && cd ../ && \
    cd ./optimum-habana/examples/text-generation/ && \
    pip install -r requirements.txt && \
    cd / && \
    pip install peft einops datasets && pip list && \
    git clone https://github.com/HabanaAI/DeepSpeed.git && \
    cd DeepSpeed && \
    git checkout -b v1.13 origin/1.13.0 && \
    pip install -e .

# Install dependency
RUN cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/examples/finetuning/instruction && \
    sed -i '/find-links https:\/\/download.pytorch.org\/whl\/torch_stable.html/d' requirements.txt && \
    sed -i '/^torch/d;/^intel-extension-for-pytorch/d' requirements.txt && \
    pip install -r requirements.txt && \
    cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/ && \
    pip install -r requirements_hpu.txt && \
    pip install transformers==4.34.1 && \
    pip install accelerate==0.24.0 && \
    pip install datasets==2.14.7

WORKDIR /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/

## NVIDIA GPU environment
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 as nvgpu

ARG ITREX_VER=main
ARG PYTHON_VERSION=3.10
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git
ARG REPO_PATH=""

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install system dependencies
SHELL ["/bin/bash", "--login", "-c"]
RUN apt update \
    && apt install -y build-essential \
    && apt install -y wget numactl git nvidia-cuda* \
    && apt install -y openssh-server \
    && apt install -y python${PYTHON_VERSION} python3-pip \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# Download ITREX code
RUN mkdir -p /intel-extension-for-transformers
COPY ${REPO_PATH} /intel-extension-for-transformers
RUN if [ "$REPO_PATH" == "" ]; then rm -rf intel-extension-for-transformers/* && rm -rf intel-extension-for-transformers/.* ; git clone --single-branch --branch=${ITREX_VER} ${REPO} intel-extension-for-transformers ; fi
WORKDIR /intel-extension-for-transformers

RUN pip install -r /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/examples/finetuning/instruction/requirements.txt
RUN pip install -r /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/requirements.txt
RUN cd /intel-extension-for-transformers && sed -i '/^torch==/d' requirements.txt && pip install -r requirements.txt && python setup.py install

WORKDIR /intel-extension-for-transformers

CMD ["/usr/sbin/sshd", "-D"]
