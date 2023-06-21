# Setup Conda
```shell
wget  https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
bash `Miniconda*.sh`
source ~/.bashrc
```

# Install numactl
```shell
apt install numactl
```

# Install Python dependencies
```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y

conda install jemalloc gperftools -c conda-forge -y

conda install pytorch torchvision cpuonly -c pytorch

pip install intel_extension_for_pytorch

pip install SentencePiece peft evaluate nltk datasets

pip install transformers diffusers accelerate intel_extension_for_transformers

pip install fastapi uvicorn sse_starlette
```

# Install git-lfs
```shell
conda install git-lfs
git-lfs install
```

# Get the LLM model
You have the option to download either the official llama-7b, mpt-7b or gpt-j-6b model. Additionally, you can contact us via email(inc.maintainers@intel.com) to acquire the optimized models.
```shell
git clone https://huggingface.co/decapoda-research/llama-7b-hf
```
or

```shell
git clone https://huggingface.co/mosaicml/mpt-7b
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