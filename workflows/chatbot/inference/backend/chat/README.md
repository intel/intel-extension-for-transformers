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
conda install numpy
conda install filelock
conda install sympy==1.12
```

Install other dependencies using pip:

```bash
pip install fastapi
pip install intel_extension_for_pytorch
pip install SentencePiece peft evaluate nltk datasets
pip install transformers
pip install diffusers accelerate
pip install fastapi uvicorn sse_starlette einops
pip install python-multipart
pip install gptcache
pip install intel_extension_for_transformers
pip install speechbrain
pip3 install torch torchvision torchaudio
pip install soundfile
pip install sentence-transformers
pip install --no-cache-dir -U chardet
pip install requests
pip install --no-cache-dir -U urllib3
pip install --force-reinstall charset-normalizer==3.2.0
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

```shell
cd intel-extension-for-transformers/workflows/chatbot/inference/backend/llmcache
git clone https://huggingface.co/hkunlp/instructor-large
```

# Modify run_ipex.sh/run_itrex.sh
Modify the model path in run scripts.

# Run the Chat server
```shell
cd intel-extension-for-transformers/workflows/chatbot/inference
nohup bash backend/chat/run_ipex.sh &
```

or 

```shell
cd intel-extension-for-transformers/workflows/chatbot/inference
nohup bash backend/chat/run_itrex.sh &
```

# Install dependencies for TalkingBot

```
pip install speechbrain
pip install soundfile
pip install pydub
```
