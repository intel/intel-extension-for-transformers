This README serves as a guide to set up the backend for a code generation chatbot utilizing the NeuralChat framework. You can deploy this code generation chatbot across various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU, and Client GPU.

This code generation chatbot demonstrates how to deploy it specifically on Intel XEON processors using Intel(R) Tensor Processing Primitives extension for PyTorch. To run the 34b or 70b level LLM model, we require implementing model parallelism using multi-node strategy.


# Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash `Miniconda*.sh`
source ~/.bashrc
```

# Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

# Install Python Dependencies

Install Intel(R) Tensor Processing Primitives extension for PyTorch from source code.

```shell
# Install from source code
git clone https://github.com/libxsmm/tpp-pytorch-extension.git
cd tpp-pytorch-extension/
git checkout feature_mxfp4_poc
git submodule update --init
python setup.py install
```

Install Intel MPI:
```shell
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/749f02a5-acb8-4bbb-91db-501ff80d3f56/l_mpi_oneapi_p_2021.12.0.538_offline.sh
bash l_mpi_oneapi_p_2021.12.0.538_offline.sh
torch_ccl_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))" 2> /dev/null)
source $torch_ccl_path/env/setvars.sh
```

Install OneCCL for multi-node model
```shell
# Specify the torch-ccl version according to your torch version (v2.1.0 in this example)
# Check out the corresponding version here: https://github.com/intel/torch-ccl/
git clone -b ccl_torch2.1.0+cpu https://github.com/intel/torch-ccl.git torch-ccl-2.2.0
cd torch-ccl-2.2.0
git submodule sync
git submodule update --init --recursive
python setup.py install
```

Install neuralchat dependencies:

```bash
pip install -r ../../../requirements.txt
pip install transformers==4.31.0 # need to downgrade transformers to 4.31.0 for LLAMA
```

Currently there are some issues when using BF16, so we need to enable MXFP4 by the below commands:
```bash
export TPP_CACHE_REMAPPED_WEIGHTS=0 
export USE_MXFP4=1
```

# Configure Multi-node
To use the multi-node model parallelism with Xeon servers, you need to configure a hostfile first and make sure ssh is able between your servers.

For example, you have two servers which have the IP of `192.168.1.1` and `192.168.1.2`, and each of it has 4 CPUs.

## Modify hostfile
```shell
vim ../../../server/config/hostfile
    192.168.1.1 slots=4
    192.168.1.2 slots=4
```

## Configure SSH between Servers
In order to enable this hostfile, you have to make sure `192.168.1.1` and `192.168.1.2` are able to access each other via SSH. Check it with `ssh 192.168.1.2`.

If your servers are not available with SSH, follow the instructions below.

1. Generate SSH Key
    Execute this command on both servers to generate ssh key in  `~/.ssh/`.
    ```shell
    ssh-keygen
    ```
2. Copy SSH Key
    Execute this command on both servers. Specify your user on the server.
    ```shell
    ssh-copy-id user@192.168.1.2
    ```
3. Test SSH Connection
    Test whether SSH is available now.
    ```shell
    ssh user@192.168.1.2
    ```
4. Check Network
    Check network communication between servers.
    ```shell
    ping 192.168.1.2
    ```

If you cannot SSH to your local server via IP, configure it with localhost as below.
```shell
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```
Then modify hostfile as below.
```shell
localhost slots=4
192.168.1.2 slots=4
```

# Configure the codegen.yaml

You can customize the configuration file 'codegen.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

|  Item              | Value                                      |
| ------------------- | --------------------------------------- |
| host                | 0.0.0.0                              |
| port                | 8000                                   |
| model_name_or_path  | "Phind/Phind-CodeLlama-34B-v2"        |
| device              | "cpu"                                  |
| use_deepspeed       | true                                  |
| world_size          | 8                                      |
| tasks_list          | ['codegen']                           |


# Run the Code Generation Chatbot Server
Before running the code-generating chatbot server, make sure you have already deploy the same `conda environment` and `intel-extension-for-tranformers codes` on both servers.

To start the code-generating chatbot server, use the following command:

```shell
nohup python run_code_gen.py &
```
