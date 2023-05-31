# Setup conda
- wget  https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
- bash `Miniconda*.sh`
- source ~/.bashrc

# Install numactl

- sudo apt install numactl

# Install python dependencies

- conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y

- conda install jemalloc gperftools -c conda-forge -y

- conda install pytorch torchvision cpuonly -c pytorch-nightly

- pip install farm-haystack==1.14.0

- pip install SentencePiece peft evaluate nltk datasets

- pip install transformers diffusers accelerate intel_extension_for_transformers

- pip install fastapi uvicorn sse_starlette

# Install elasticsearch
- su ubuntu  # run elasticsearch with non-root user
- sudo apt update
- sudo apt install default-jre
- sudo apt install default-jdk
- wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-linux-x86_64.tar.gz
- tar -xvzf elasticsearch-7.17.10-linux-x86_64.tar.gz
- cd elasticsearch-7.17.10/
- ./bin/elasticsearch &

### Install git-lfs
- conda install git-lfs

- git-lfs install

### Download model
You can download the official llama-7b model. Additionally, you can contact us via mail(inc.maintainers@intel.com) to acquire the optimized models.

# Modify run.sh
Modify the model path in run.sh.

# Run the fastrag server
- nohup bash run.sh &

