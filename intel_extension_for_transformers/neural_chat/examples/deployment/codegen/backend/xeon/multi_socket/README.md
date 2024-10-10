This README serves as a guide to set up the backend for a code generation chatbot utilizing the NeuralChat framework. You can deploy this code generation chatbot across various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU, and Client GPU.

This code generation chatbot demonstrates how to deploy it specifically on Intel XEON processors. To run the 34b or 70b level LLM model, we require implementing model parallelism using multi-node strategy.


# Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash `Miniconda*.sh`
source ~/.bashrc
# Create Conda Virtual Environment
conda create -n your_env_name python=3.10 -y
conda activate your_env_name
```


# Install Python Dependencies

Install Dependencies from conda & pip
```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y

pip install transformers
```

Install DeepSpeed from source code
```shell
# git clone Deepspeed source code
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
# cherry-pick commits
git remote add DeepSpeedSYCLSupport https://github.com/delock/DeepSpeedSYCLSupport.git
git fetch DeepSpeedSYCLSupport
git cherry-pick cd070bf8
# cherry-pick commits
git remote add ftian1 https://github.com/ftian1/DeepSpeed.git
git fetch ftian1
git log ftian1/master --pretty=format:"%h" -4
git cherry-pick a3ce24dc
git cherry-pick 6d3174a2
git cherry-pick 7fc67306
# install DeepSpeed from source code
pip install .
```

Install OneCCL
```shell
git clone https://github.com/oneapi-src/oneCCL.git
cd oneCCL
mkdir build
cd build
cmake ..
make -j install
source _install/env/setvars.sh
python -m pip install oneccl_bind_pt --index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

Install IPEX CPU version:

```shell
python -m pip install intel_extension_for_pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

# Run the Code Generation Chatbot Server

```shell
deepspeed --bind_cores_to_rank ./run_multi_socket.py
```
