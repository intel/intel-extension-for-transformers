Step-by-Step
=========
This document describes the end-to-end workflow for Huggingface model [VIT](https://huggingface.co/google/vit-base-patch16-224) with Neural Engine backend.

# Prerequisite

## 1. Installation
### 1.1 Install python environment
Create a new python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```
Make sure you have the autoconf installed. 
Also, `gcc` higher than 9.0, `cmake` higher than 3 is required.
```shell
gcc -v
cmake --version
conda install cmake
sudo apt install autoconf
```
Install Intel® Extension for Transformers, please refer to [installation](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md)
```shell
# Install from pypi
pip install intel-extension-for-transformers

# Install from source code
cd <intel_extension_for_transformers_folder>
git submodule update --init --recursive
python setup.py install
```
Install required dependencies for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/sst2/bert_mini
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11


### 1.2 Environment variables preload libjemalloc.so can improve the performance when multi instances.
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instances.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
## 2. Prepare Dataset and Pretrained Model

### 2.1 Get Dataset

```shell
python prepare_dataset.py --task_name=image-classification --output_dir=./data
```

### 2.2 Get model
Neural Engine can parse Tensorflow/Pytorch/ONNX model and Neural Engine IR.
You can get a fp32 ONNX model from the optimization module by setting precision=fp32, command as follows:
```shell
bash prepare_model.sh --input_model=google/vit-base-patch16-224  --task_name=imagenet-1k --output_dir=./model_and_tokenizer --precision=fp32
```
By setting precision=int8, you can get a PTQ int8 model or setting precision=bf16 to get a bf16 model.
```shell
bash prepare_model.sh --input_model=google/vit-base-patch16-224  --task_name=imagenet-1k --output_dir=./model_and_tokenizer --precision=int8
```
Note: the input_model could be changed from a vit base model to a vit large model.

## Benchmark
Throught setting --dynamic_quanzite for FP32 model, you could benchmark dynamic quantize int8 model.
### 2.1 accuracy  
Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=accuracy --data_dir=./data --batch_size=8
  ```

### 2.2 Performance
Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=8 --seq_len=128
  ```

  Or compile framwork model to IR using python API
  ```
  from intel_extension_for_transformers.backends.neural_engine.compile import compile
  graph = compile('./model_and_tokenizer/int8-model.onnx')
  graph.save('./ir')
  ```
