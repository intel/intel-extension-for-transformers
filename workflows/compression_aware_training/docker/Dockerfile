FROM intel/intel-optimized-pytorch:pip-ipex-1.13.100-ubuntu-22.04

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev 

RUN mkdir -p /workspace/output

COPY . /workspace

WORKDIR /workspace

RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt
