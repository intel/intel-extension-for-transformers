### Setup conda
- wget  https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
- bash `Miniconda*.sh`
- source ~/.bashrc

### Install numactl

- apt install numactl


### Install python dependencies
- conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y

- conda install jemalloc gperftools -c conda-forge -y

- pip install torch==1.13.1
- pip install torchvision==0.14.1
- pip install intel_extension_for_pytorch==1.13.0

- pip install transformers diffusers bottle gevent accelerate

- pip install pymysql



### Install git-lfs
- conda install git-lfs

- git-lfs install

### Download model
- git clone https://huggingface.co/runwayml/stable-diffusion-v1-5


# Run Stable Diffusion Server
- nohup bash run.sh &
