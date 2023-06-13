Step-by-Step
=========
This document describes the end-to-end workflow for Huggingface model [VIT](https://huggingface.co/google/vit-base-patch16-224) with Neural Engine backend.

# Prerequisite

## Installation
### Install python environment
Create a python environment, optionally with autoconf for jemalloc support.
```shell
conda create -n <env name> python=3.8 [autoconf]
conda activate <env name>
```

Check that `gcc` version is higher than 9.0.
```shell
gcc -v
```

Install Intel® Extension for Transformers, please refer to [installation](/docs/installation.md).
```shell
# Install from pypi
pip install intel-extension-for-transformers

# Or, install from source code
cd <intel_extension_for_transformers_folder>
pip install -v .
```

Install required dependencies for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/huggingface/pytorch/image-classification/deployment/imagenet/vit
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11


### Environment Variables (Optional)
```shell
# Preload libjemalloc.so may improve the performance when inference under multi instance.
conda install jemalloc==5.2.1 -c conda-forge -y
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so

# Using weight sharing can save memory and may improve the performance when multi instances.
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
>**Note**: This step is optional.

## Inference Pipeline
Neural Engine can parse ONNX model and Neural Engine IR.
We provide with three `mode`s: `accuracy`, `throughput` or `latency`. For throughput mode, we will use multi-instance with 4cores/instance occupying one socket.
You can run fp32 model inference by setting `precision=fp32`, command as follows:
```shell
bash run_vit.sh --model=google/vit-base-patch16-224  --dataset=imagenet-1k --precision=fp32 --mode=throughput
```
By setting `precision=int8` you could get PTQ int8 model and setting `precision=bf16` to get bf16 model.
```shell
bash run_vit.sh --model=google/vit-base-patch16-224  --dataset=imagenet-1k --precision=int8 --mode=throughput
```
Note: the input_model could be changed from a vit base model to a vit large model.

## Benchmark
If you want to run local onnx model inference, we provide with python API and C++ API. To use C++ API, you need to transfer to model ir fisrt.
### Accuracy  

Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=accuracy --data_dir=path-to-dataset --batch_size=1 --warm_up=100 --iteration=1000
  ```

### Performance
Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=1 --warm_up=100 --iteration=1000
  ```

