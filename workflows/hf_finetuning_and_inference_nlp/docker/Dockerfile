# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

ARG IMAGE_NAME=intel/ai-tools
ARG IMAGE_TAG=itrex-0.1.1
FROM ${IMAGE_NAME}:${IMAGE_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    git \
    python3-dev 

RUN mkdir -p /workspace/output

COPY . /workspace

WORKDIR /workspace

RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt
