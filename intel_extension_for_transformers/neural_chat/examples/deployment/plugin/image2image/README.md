
This README is intended to guide you through setting up the service of image2image using the NeuralChat framework. You can deploy it on Intel XEON Scalable Processors.

# Introduction
NeuralChat provides services not only based on LLM, but also support single service of plugin such as image2image. This example introduces our solution to build a plugin-as-service server. Though few lines of code, our api could help the user build a image2image plugin service. This plugin leverages the Neural Engine backend to speed up the image2image inference speed. We can get a high quality image only taking about 5 seconds on Intel XEON SPR server.

# Setup Environment

## Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
```

## Prepare Python Environment
Create a python environment, optionally with autoconf for jemalloc support.
```shell
conda create -n image2image python=3.9.0
conda activate image2image
```
>**Note**: Make sure pip <=23.2.2

Check that `gcc` version is higher than 9.0.
```shell
gcc -v
```

## Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

## Install Python dependencies

Install the following Python dependencies using Conda:

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
```

Install Intel® Extension for Transformers, please refer to [installation](/docs/installation.md).
```shell
# Install from pypi
pip install intel-extension-for-transformers
```

Install other dependencies using pip:

```shell
pip install -r requirements.txt
pip install -r ../../../../requirements.txt
pip install transformers==4.28.1
pip install diffusers==0.12.1
```
>**Note**: Please use transformers no higher than 4.28.1

# Prepare Stable Diffusion Models

Export FP32 ONNX models from the hugginface diffusers module, command as follows:

```shell
python prepare_model.py --input_model=timbrooks/instruct-pix2pix --output_path=./model_pix2pix --bf16
```

# Compile Models

Export three BF16 onnx sub models of the stable diffusion to Nerual Engine IR.

```shell
bash export_model.sh --input_model=model_pix2pix --precision=bf16
```



# Run the image2image service server
To start the image2image service server, run the following command:

```shell
nohup bash run.sh &
```

# Call the image2image plugin service
To call the started image2image service, the API is listed as follows:
http://127.0.0.1:8000/plugin/image2image
You can modify the IP address and port number based on your requirements.
