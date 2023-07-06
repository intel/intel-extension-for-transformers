
## SPR environment
ARG UBUNTU_VER=22.04
FROM ubuntu:${UBUNTU_VER} as cpu

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

WORKDIR /root/chatbot

# Install system dependencies
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
SHELL ["/bin/bash", "--conda run", "-c"]
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash && \
    unset -f conda && \
    export PATH=$CONDA_DIR/bin/:${PATH} && \
    conda config --add channels intel && \
    # conda install python==3.9 \
    conda create -yn chatbot-demo python=3.9 && \
    echo "conda activate chatbot-demo" >> ~/.bashrc

SHELL ["conda", "run", "-n", "chatbot-demo", "/bin/bash", "--login", "-c"]

RUN conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y && \
    conda install jemalloc gperftools -c conda-forge -y && \
    conda install pytorch torchvision torchaudio cpuonly -c pytorch && \
    pip install farm-haystack==1.14.0 && \
    pip install intel_extension_for_pytorch && \
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
