### Setup conda

First, install and configure the Conda environment:

```bash
wget  https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
bash `Miniconda*.sh`
source ~/.bashrc
```

### Install numactl

Install the numactl library:

```bash
apt install numactl
```

### Install python dependencies

Install the following Python dependencies using Conda:

```bash
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
```

Install other dependencies using pip:

```bash
pip install torch==1.13.1
pip install torchvision==0.14.1
pip install intel_extension_for_pytorch==1.13.0
pip install transformers diffusers bottle gevent accelerate
pip install pymysql
```


### Install git-lfs

Install Git LFS for downloading model.

```
conda install git-lfs
git-lfs install
```

### Download model

You can download the official [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) model. Additionally, you can contact us via mail(inc.maintainers@intel.com) to acquire the optimized models.

```bash
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

# Run Stable Diffusion Server

Start the Stable Diffusion server:

```bash
nohup bash run.sh &
```