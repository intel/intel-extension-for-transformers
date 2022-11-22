# Sparse model Step-by-Step
Here is a example from pruning a distilbert base model using group lasso during a distillation process to get sparse model, and then 
inference with SparseLib which is a high-performance operator computing library. Overall, get performance improvement.
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
cd <intel_extension_for_transformers_folder>
git submodule update --init --recursive
python setup.py install
```
Install package for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/sparse/distilbert_base_uncased
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

### 2\. Prepare Dataset and pretrained model

### 2.1 Get dataset

```shell
python prepare_dataset.py --dataset_name=squad --output_dir=./data
```

### 2.2 Get sparse model

Use the [sparse model](https://huggingface.co/Intel/distilbert-base-uncased-squadv1.1-sparse-80-1x4-block-pruneofa) we publiced on huggingface which is distilbert base on Squad1.1 with sparse ration 80 on 1X4 block.
You can get INT8 ONNX sparse model from optimization module by setting precision=int8, command as follows:
```shell
bash prepare_model.sh --input_model=Intel/distilbert-base-uncased-squadv1.1-sparse-80-1x4-block-pruneofa  --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=int8
```
Then you can generate tranposed sparse model to get better performance, command as follows:
```shell
python export_tranpose_ir.py --input_model=./model_and_tokenizer/int8-model.onnx
```

### Benchmark
Neural Engine will automatically detect weight structured sparse ratio, as long as it beyond 70% (since normaly get performance gain when sparse ratio beyond 70%), Neural Engine will call [SparseLib](https://github.com/intel/intel-extension-for-transformers/tree/develop/intel_extension_for_transformers/backends/neural_engine/SparseLib) kernels and high performance layernorm op with transpose mode to improve inference performance.

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
  
  Or run C++
  The warmup below is recommended to be 1/10 of iterations and no less than 3.
  
  ```
  export GLOG_minloglevel=2
  export OMP_NUM_THREADS=<cpu_cores>
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  export UNIFIED_BUFFER=1
  numactl -C 0-<cpu_cores-1> <intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/bin/neural_engine
  --batch_size=<batch_size> --iterations=<iterations> --w=<warmup>
  --seq_len=128 --config=./sparse_int8_ir/conf.yaml --weight=./sparse_int8_ir/model.bin
  ```
