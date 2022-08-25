# Sparse model Step-by-Step
Here is a example from pruning a bert mini model using group lasso during a distillation process to get sparse model, and then 
inference with SparseLib which is a high-performance operator computing library. Overall, get performance and accuracy improvement.
# Prerequisite

### 1\. Installation

1.1 Install python environment
Create a new python environment

```shell
conda create -n <env name> python=3.8
conda activate <env name>
```

Check the gcc version using $gcc-v, make sure the gcc version is higher than 7.0.
If not, you need to update gcc by yourself.
Make sure the cmake version is 3 rather than 2.
Make sure you have the autoconf installed.
If not, you need to install autoconf by yourself.
If not, you need to install cmake.

```shell
cmake --version
conda install cmake
sudo apt install autoconf
```

Install NLPTookit from source code

```shell
cd <NLP_Toolkit_folder>
git submodule update --init --recursive
python setup.py install
```
Install package for examples
```shell
cd <NLP_Toolkit_folder>/examples/deployment/neural_engine/sst2/bert_mini
pip install -r requirements.txt
```

1.2 Environment variables Preload libjemalloc.so can improve the performance when multi instance.

```
export LD_PRELOAD=<NLP_Toolkit_folder>/nlp_toolkit/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```

Using weight sharing can save memory and improve the performance when multi instance.

```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```

### 2\. Prepare Dataset and pretrained model

### 2.1 Get dataset

```shell
python prepare_dataset.py --dataset_name=glue --task_name=sst2 --output_dir=./data
```

### 2.2 Get sparse model

Neural Engine can parse Sparse ONNX model and Neural Engine IR.
You can train a Bert mini sst2 sparse model with distillation through Neural Compressor [example](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/28e9b1e66c23f4443a2be8f2926fee1e919f5a14/examples/pytorch/nlp/huggingface_models/text-classification/pruning_while_distillation/group_lasso/eager/README.md). and transpose the weight and activation to get better performance.
Neural Engine will automatically detect weight structured sparse ratio, as long as it beyond 70% (since normaly get performance gain when sparse ratio beyond 70%), Neural Engine will call [SparseLib](https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit/tree/develop/nlp_toolkit/backends/neural_engine/SparseLib) kernels and high performance layernorm op with transpose mode to improve inference performance.

### Benchmark

  2.1 accuracy
  run python
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./sparse_int8_ir  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=8
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --input_model=./sparse_int8_ir  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=8
  ```

  2.2 performance
  run python
  
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./sparse_int8_ir --mode=performance --batch_size=8 --seq_len=128
  ```
  
  or run shell
  
  ```shell
  bash run_benchmark.sh --input_model=./sparse_int8_ir  --mode=performance --batch_size=8 --seq_len=128
  ```
  
  or compile framwork model to IR using python API
  
  ```
  from nlp_toolkit.backends.neural_engine.compile import compile
  graph = compile('./sparse_int8_ir')
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
  --seq_len=128 --config=./ir/conf.yaml --weight=./ir/model.bin
  ```
