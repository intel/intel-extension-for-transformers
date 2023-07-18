# Setup conda

First, install and configure the Conda environment:

```bash
wget  https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
bash `Miniconda*.sh`
source ~/.bashrc
```

# Install numactl

Install the numactl library:

```bash
sudo apt install numactl
```

# Install python dependencies

Install the following Python dependencies using Conda:

```bash
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
conda install pytorch torchvision cpuonly -c pytorch-nightly
```

Install other dependencies using pip:

```bash
pip install farm-haystack==1.14.0
pip install intel_extension_for_pytorch
pip install SentencePiece peft evaluate nltk datasets
pip install transformers diffusers accelerate intel_extension_for_transformers
pip install fastapi uvicorn sse_starlette einops PyPDF2 chromadb langchain openpyxl InstructorEmbedding
```

# Install elasticsearch

Run Elasticsearch as a non-root user:

```bash
su ubuntu  # run elasticsearch with non-root user
sudo apt update
sudo apt install default-jre
sudo apt install default-jdk
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-linux-x86_64.tar.gz
tar -xvzf elasticsearch-7.17.10-linux-x86_64.tar.gz
cd elasticsearch-7.17.10/
nohup ./bin/elasticsearch &
```

### Install git-lfs

Install Git LFS for downloading model.

```bash
conda install git-lfs
git-lfs install
```

### Download model
You can download the official [mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat) model. Additionally, you can contact us via mail(inc.maintainers@intel.com) to acquire the optimized models.

# Modify run.sh
Modify the model path in run.sh.

# Run the fastrag server

Start the FastRAG server:

```bash
nohup bash run.sh &
```
