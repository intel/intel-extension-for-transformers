#!/bin/bash

ENV_NAME=env_itrex_gpu
conda deactivate
echo "conda env remove -n $ENV_NAME"
conda env remove -n $ENV_NAME
echo "conda create -n $ENV_NAME python=3.9 -y"
conda create -n $ENV_NAME python=3.9 -y
echo "conda activate  $ENV_NAME"
conda activate  $ENV_NAME

pip install --upgrade pip
pip install -r requirements-gpu.txt
pip uninstall torch torchvision -y
pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
echo "Run environment $ENV_NAME is created"
echo "conda activate $ENV_NAME"
echo "Build and install ITREX in $ENV_NAME"
pip install -v .
echo "conda activate $ENV_NAME"
