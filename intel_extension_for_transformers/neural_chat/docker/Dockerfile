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
    source ~/.bashrc

RUN source activate && conda activate neuralchat && pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable-cpu && \
    cd /intel-extension-for-transformers && pip install -r requirements.txt && pip install -v . && \
    cd ./intel_extension_for_transformers/neural_chat/examples/instruction_tuning && pip install -r requirements.txt && \
    cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat && pip install -r requirements_cpu.txt && \
    conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi future six requests dataclasses -y && \
    conda install jemalloc gperftools -c conda-forge -y && \
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
FROM vault.habana.ai/gaudi-docker/1.11.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest as hpu

ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/root:/usr/lib/habanalabs/

RUN apt-get update && \
    apt-get install git-lfs && \
    git-lfs install

COPY inference/0001-fix-lm-head-overriden-issue-move-it-from-checkpoint-.patch /

RUN git clone https://github.com/huggingface/optimum-habana.git && \
    cd optimum-habana/ && git reset --hard b6edce65b70e0fadd5d5f51234700bd1144cd0b0 && pip install -e . && cd ../ && \
    cd ./optimum-habana/examples/text-generation/ && \
    pip install -r requirements.txt && \
    cd / && \
    pip install peft && \
    pip install einops && \
    pip install datasets && \
    git clone https://github.com/HabanaAI/DeepSpeed.git && \
    cd DeepSpeed && \
    git checkout -b v1.11 origin/1.11.0 && \
    git apply /0001-fix-lm-head-overriden-issue-move-it-from-checkpoint-.patch && \ 
    pip install -e .

# Download ITREX code
ARG ITREX_VER=main
ARG REPO=https://github.com/intel/intel-extension-for-transformers.git

RUN git clone --single-branch --branch=${ITREX_VER} ${REPO} intel-extension-for-transformers && \
    cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/examples/instruction_tuning/ && \
    pip install -r requirements.txt && \
    cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/ && \
    pip install -r requirements_hpu.txt

# Build ITREX
RUN cd /intel-extension-for-transformers && pip install -r requirements.txt && pip install -v . && \
    pip install transformers==4.32.0 && \
    pip install accelerate==0.22.0 && \
    pip install intel_extension_for_deepspeed intel_extension_for_pytorch

WORKDIR /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/

