# Step-by-Step

# Prerequisite

### 1. Installation
1.1 Install python environment
Create a new python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```
Check the gcc version using $gcc-v, make sure the gcc version is higher than 9.0.
If not, you need to update gcc by yourself.
Make sure you have the autoconf installed.
If not, you need to install autoconf by yourself.
Make sure the cmake version is 3 rather than 2.
If not, you need to install cmake.
```shell
cmake --version
conda install cmake
sudo apt install autoconf
```
Install Intel Extension for Transformers from Source Code
```shell
cd <intel_extension_for_transformers_folder>
git submodule update --init --recursive
python setup.py install
```
Install package for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/squad/bert_large
pip install -r requirements.txt
```
1.2 Environment variables Preload libjemalloc.so can improve the performance when multi instance.
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instance.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
### 2. Prepare Dataset and pretrained model

### 2.1 Get dataset

```shell
python prepare_dataset.py --dataset_name=squad --output_dir=./data
```

### 2.2 Get Model
You can get FP32 pytorch model from optimization module by setting precision=fp32, command as follows:
```shell
bash prepare_model.sh --input_model=bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=fp32
```

Throught setting precision=int8 you could get PTQ int8 JIT file
```
bash prepare_model.sh --input_model=bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=int8
```

### Benchmark
  2.1 accuracy  

  run python
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1
  ```
  if you just want a quick start, you can run only a part of dataset, like this
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1 --max_eval_samples=10
  ```
  but the accuracy of quick start is unauthentic.

  2.2 performance

  run python
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=performance --batch_size=1 --seq_len=384
  ```