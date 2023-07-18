# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

FROM intel/ai-tools:itrex-0.1.1

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev 

RUN mkdir -p /workspace/output

COPY . /workspace

WORKDIR /workspace

RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt
