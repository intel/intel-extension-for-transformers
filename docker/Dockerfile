# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

ARG IMAGE_NAME=ubuntu
ARG IMAGE_TAG=22.04
FROM ${IMAGE_NAME}:${IMAGE_TAG} as base

FROM base as itrex-base

ARG PYTHON=python3.10

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ${PYTHON} \
    python3-pip && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf $(which ${PYTHON}) /usr/bin/python

RUN ${PYTHON} -m pip install -U pip

FROM itrex-base as devel

ENV DEBIAN_FRONTEND=noninteractive

ENV LANG C.UTF-8
ARG PYTHON=python3.10

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    autoconf \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ${PYTHON}-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/intel-extension-for-transformers
WORKDIR /app/intel-extension-for-transformers
COPY . /app/intel-extension-for-transformers

RUN ${PYTHON} -m pip install -r requirements.txt --no-cache-dir -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    ${PYTHON} -m pip install -r tests/requirements.txt --no-cache-dir -f https://developer.intel.com/ipex-whl-stable-cpu && \
    ${PYTHON} -m pip install . --no-cache-dir && \
    rm -rf .git*

FROM itrex-base as prod

ARG PYTHON=python3.10

COPY --from=devel /usr/local/lib/${PYTHON}/dist-packages /usr/local/lib/${PYTHON}/dist-packages
COPY --from=devel /usr/local/bin /usr/local/bin
COPY --from=devel /app/intel-extension-for-transformers/docker /app/licensing
