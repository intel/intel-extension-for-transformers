# Step-by-Step

# Prerequisite

### 1. Installation
1.1 Install python environment
Create a new python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```
Check the gcc version using $gcc-v, make sure the gcc version is higher than 7.0.
If not, you need to update gcc by yourself.
Make sure the cmake version is 3 rather than 2.
If not, you need to install cmake.
```shell
cmake --version
conda install cmake
```
Install NLPTookit from source code
```shell
cd <NLP_Toolkit_folder>
git submodule update --init --recursive
python setup.py install
```
Install package for examples
```shell
cd <NLP_Toolkit_folder>/examples/deployment/neural_engine/squad/bert_large
pip install -r requirements.txt
```
1.2 Environment variables
Preload libiomp5.so can improve the performance when bs=1.
```
export LD_PRELOAD=<path_to_libiomp5.so>
```
Preload libjemalloc.so can improve the performance when multi instance.
```
export LD_PRELOAD=<NLP_Toolkit_folder>/nlp_toolkit/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instance.
```
export SHARED_INST_NUM=<inst_num>
```
### 2. Prepare Dataset and pretrained model

### 2.1 Get dataset

```shell
python prepare_dataset.py --dataset_name=squad --output_dir=./data
```

### 2.2 Get model
Neural_Engine can parse Tensorflow/Pytorch/ONNX and IR model.
Here are two examples to get ONNX model.
You can get FP32 modol from optimize by setting precision=fp32 as follows:
```shell
bash prepare_model.sh --input_model=bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=fp32
```
And for better perfromance, you can also get a PTQ int8 model by setting tune.
```shell
bash prepare_model.sh --input_model=bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=int8
```

### Benchmark

  2.1 accuracy  
  run python
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=accuracy --data_dir=./data --batch_size=1
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx --mode=accuracy --data_dir=./data --batch_size=1
  ```
  if you just want a quick start, you can run only a part of dataset, like this
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --mode=accuracy --data_dir=./data --batch_size=1 --max_eval_samples=10
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx --mode=accuracy --data_dir=./data --batch_size=1 --max_eval_samples=10
  ```
  but the accuracy of quick start is unauthentic.

  2.2 performance
  run python
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=1 --seq_len=384
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=1 --seq_len=384
  ```
  or compile framwork model to IR using python API
  ```
  from nlp_toolkit.backends.neural_engine.compile import compile
  graph = compile('./model_and_tokenizer/int8-model.onnx')
  graph.save('./ir')
  ```
  and run C++  
  The warmup below is recommended to be 1/10 of iterations and no less than 3.
  ```
  export GLOG_minloglevel=2
  export OMP_NUM_THREADS=<cpu_cores>
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  export UNIFIED_BUFFER=1
  numactl -C 0-<cpu_cores-1> <NLP_Toolkit_folder>/nlp_toolkit/backends/neural_engine/bin/neural_engine
  --batch_size=<batch_size> --iterations=<iterations> --w=<warmup>
  --seq_len=384 --config=./ir/conf.yaml --weight=./ir/model.bin
  ```
