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
#   docker build ./ -f Dockerfile -t chatbot_infer:latest
# If you need to use proxy, please use the following command
#   docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f Dockerfile -t chatbot_infer:latest

## SPR environment
ARG UBUNTU_VER=22.04
FROM ubuntu:${UBUNTU_VER} as cpu

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

WORKDIR /root/chatbot

# Install system dependencies
SHELL ["/bin/bash", "--login", "-c"]
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget numactl git \
    && apt-get clean \
    && apt-get install git \
    && rm -rf /var/lib/apt/lists/*

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
    # conda install python==3.9 \
    conda create -yn chatbot-demo python=3.9 && \
    echo "conda activate chatbot-demo" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c", "conda", "run", "-n", "chatbot-demo"]

RUN conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y && \
    conda install jemalloc gperftools -c conda-forge -y && \
    conda install pytorch torchvision torchaudio cpuonly -c pytorch && \
    pip install farm-haystack==1.14.0 && \
    pip install intel_extension_for_pytorch && \
    pip install optimum-intel && \
    pip install transformers diffusers accelerate SentencePiece peft evaluate nltk datasets && \
    pip install fastapi uvicorn sse_starlette bottle gevent pymysql && \
    pip install schema && \
    pip install -i https://test.pypi.org/simple/ intel-extension-for-transformers==1.0.0.dev20230602 && \
    pip install datasets torch transformers sentencepiece peft evaluate nltk rouge_score


RUN cd /root/chatbot && git clone https://github.com/intel/intel-extension-for-transformers.git \
    && cd ./intel-extension-for-transformers/workflows/chatbot/inference/ && pip install -r requirements.txt

WORKDIR /root/chatbot/intel-extension-for-transformers/workflows/chatbot/inference/


# HABANA environment
FROM vault.habana.ai/gaudi-docker/1.10.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest as hpu

ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/root:/usr/lib/habanalabs/

RUN git clone https://github.com/huggingface/optimum-habana.git && \
    cd ./optimum-habana/examples/text-generation/ && \
    pip install -r requirements.txt && \
    apt-get update && \
    apt-get install git-lfs && \
    git-lfs install

RUN pip install optimum[habana] && \
    pip install peft && \
    pip install einops && \
    pip install datasets && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.10.0

RUN git clone https://github.com/intel/intel-extension-for-transformers.git \
    && cd ./intel-extension-for-transformers/workflows/chatbot/inference/ \
    && pip install -r requirements.txt && \
    git clone https://huggingface.co/mosaicml/mpt-7b-chat

WORKDIR /intel-extension-for-transformers/workflows/chatbot/inference/
