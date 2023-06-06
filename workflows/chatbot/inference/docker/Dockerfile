ARG UBUNTU_VER=20.04
FROM ubuntu:${UBUNTU_VER} as deploy

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

WORKDIR /root/chatbot

# Install system dependencies
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget numactl git \
    && apt-get clean \
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
    conda install python==3.9

RUN pip install datasets torch transformers sentencepiece peft evaluate nltk rouge_score