# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04 as base

# Create an intermediate stage called itrex-base
FROM base as itrex-base

# Set the Python version
ARG PYTHON=python3.10

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ${PYTHON} \
    python3-pip && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link for Python
RUN ln -sf $(which ${PYTHON}) /usr/bin/python

# Upgrade pip
RUN ${PYTHON} -m pip install -U pip

# Create a development stage
FROM itrex-base as devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# Set the Python version
ARG PYTHON=python3.10

# Set up the working directory
RUN mkdir -p /app/intel-extension-for-transformers
WORKDIR /app/intel-extension-for-transformers

# Install necessary development packages
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

# Copy the entire repository content
COPY . /app/intel-extension-for-transformers

# Install required Python packages
RUN ${PYTHON} -m pip install -r requirements.txt --no-cache-dir -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN ${PYTHON} -m pip install -r tests/requirements.txt --no-cache-dir -f https://developer.intel.com/ipex-whl-stable-cpu
RUN ${PYTHON} -m pip install . --no-cache-dir && \
    rm -rf .git*

# Create a production stage
FROM itrex-base as prod

# Set the Python version
ARG PYTHON=python3.10

# Copy necessary files from the development stage to the production stage
COPY --from=devel /usr/local/lib/${PYTHON}/dist-packages /usr/local/lib/${PYTHON}/dist-packages
COPY --from=devel /usr/local/bin /usr/local/bin
