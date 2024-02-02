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
    cd /intel-extension-for-transformers && pip install -r requirements.txt && python setup.py install && \
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


ENTRYPOINT ["neuralchat_server"]
CMD ["start", "--config_file", "/codegen.yaml"]

