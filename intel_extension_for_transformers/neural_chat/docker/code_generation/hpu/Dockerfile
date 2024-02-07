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
    pip install -r requirements.txt && \
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
    pip install datasets==2.14.7 && \
    pip uninstall -y intel_extension_for_pytorch


ENTRYPOINT ["neuralchat_server"]
CMD ["start", "--config_file", "/codegen.yaml"]

