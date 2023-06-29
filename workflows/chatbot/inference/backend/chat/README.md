# Setup Conda

First, install and configure the Conda environment:

```shell
wget  https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
bash `Miniconda*.sh`
source ~/.bashrc
```

# Install numactl

Install the numactl library:

```shell
apt install numactl
```

# Install Python dependencies

Install the following Python dependencies using Conda:

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
conda install pytorch torchvision cpuonly -c pytorch
```

Install other dependencies using pip:

```bash
pip install intel_extension_for_pytorch
pip install SentencePiece peft evaluate nltk datasets
pip install transformers diffusers accelerate intel_extension_for_transformers
pip install fastapi uvicorn sse_starlette einops
```

# Install git-lfs

Install Git LFS for downloading model.

```shell
conda install git-lfs
git-lfs install
```

# Get the LLM model
You have the option to download either the official [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat) or [gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) model. Additionally, you can contact us via email(inc.maintainers@intel.com) to acquire the optimized models.

```shell
git clone https://huggingface.co/decapoda-research/llama-7b-hf
```
or

```shell
git clone https://huggingface.co/mosaicml/mpt-7b-chat
```
or

```shell
git clone https://huggingface.co/EleutherAI/gpt-j-6b
```

# Modify run_ipex.sh/run_itrex.sh
Modify the model path in run scripts.

# Run the Chat server
```shell
nohup bash run_ipex.sh &
```

or 

```shell
nohup bash run_itrex.sh &
```