# Step-by-Step

The implementation is based on [Length Adaptive Transformer](https://github.com/clovaai/length-adaptive-transformer)'s work.
Currently, it supports BERT based transformers.

[QuaLA-MiniLM: A Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114) has been accepted by NeurIPS 2022. Our quantized length-adaptive MiniLM model (QuaLA-MiniLM) is trained only once, dynamically fits any inference scenario, and achieves an accuracy-efficiency trade-off superior to any other efficient approaches per any computational budget on the SQuAD1.1 dataset (up to x8.8 speedup with <1% accuracy loss). The following shows how to reproduce this work and we also provide the [notebook tutorials](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/tutorials/pytorch/question-answering/Dynamic_MiniLM_SQuAD.ipynb).


# Prerequisite

## 1.Environment

### Prepare Python Environment
Create a python environment
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
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/squad/bert_large
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11

### Environment Variables
Preload libjemalloc.so can improve the performance when multi instance.
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instance.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
## 2.Prepare Dataset

```shell
python prepare_dataset.py --dataset_name=squad --output_dir=./data
```

## 3.Prepare Model
Neural Engine can parse ONNX model and IR.  
You could get fp32 ONNX model by setting precision=fp32, command is as follows:
```shell
bash prepare_model.sh --input_model=sguskin/dynamic-minilmv2-L6-H384-squad1.1 --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=fp32
```
By setting precision=int8 you could get PTQ int8 model and setting precision=bf16 to get bf16 model.
```shell
bash prepare_model.sh --input_model=sguskin/dynamic-minilmv2-L6-H384-squad1.1 --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=int8
```
Python API is also available:
```shell
python run_qa.py --model_name_or_path "sguskin/dynamic-minilmv2-L6-H384-squad1.1" --dataset_name squad --do_eval --output_dir model_and_tokenizer --overwrite_output_dir --length_config "(269, 253, 252, 202, 104, 34)" --overwrite_cache --to_onnx ./model_and_tokenizer/int8-model.onnx
```

You could also compile the model to IR using python API as follows:
```shell
from intel_extension_for_transformers.backends.neural_engine.compile import compile
graph = compile('./model_and_tokenizer/fp32-model.onnx')
graph.save('./ir')
```
# Benchmark
Throught setting --dynamic_quanzite for FP32 model, you could benchmark dynamic quantize int8 model.

## 1.Accuracy

Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=8
```
If you just want a quick start, you could try a small set of dataset, like this:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=accuracy --data_dir=./data --batch_size=1 --max_eval_samples=10
```
> **Note**: The accuracy of partial dataset is unauthentic.

## 2.Performance

Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=1 --seq_len=384
```
