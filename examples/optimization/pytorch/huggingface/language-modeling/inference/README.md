Step-by-Step
============
This document describes the step-by-step instructions to run large language models(LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed [Sapphire Rapids](https://www.intel.com/content/www/us/en/products/docs/processors/xeon-accelerated/4th-gen-xeon-scalable-processors.html)) with PyTorch and [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

We now support two models, and we are adding more models and more advanced techniques(distributed inference, model compressions etc.) to better unleash LLM inference on Intel platforms.

- GPT-J
  script `run_gptj.py` is based on [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) and provides inference benchmarking. For [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) quantization, please refer to [quantization example](../quantization/inc)
- BLOOM-176B
  script `run_bloom.py` is adapted from [HuggingFace/transformers-bloom-inference](https://github.com/huggingface/transformers-bloom-inference/blob/main/bloom-inference-scripts/bloom-accelerate-inference.py). 

# Prerequisite
## Create Environment
```
conda install mkl mkl-include -y
conda install jemalloc gperftools -c conda-forge -y
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch==1.13.0
pip install -r requirements.txt
```
## Setup Environment Variables
```
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# IOMP
export OMP_NUM_THREADS=< Cores number to use >
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
```

# Performance Benchmark

## GPT-J
>**Note**: Please apply ipex-patch.patch to intel-extension-for-pytorch as follow if you want to benchmark `int8` performance.
```bash
# only need to do when precision is int8
git clone https://github.com/intel/intel-extension-for-pytorch.git
git checkout v1.13.0+cpu
git submodule update --init --recursive
git apply ipex-patch.patch
python setup.py develop
```
### Text Generation
>**Note**: Please apply gen-patch.patch to transforms as follow if you want to benchmark `int8` performance.
```bash
# only need to do when precision is int8.
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 9d1116e9951686f937d17697820117636bfc05a5
git apply gen-patch.patch
python setup.py develop
```
#### Run
```bash
# use jemalloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# for generation
# default is beam search with num_beams=4, if you need to use greedy search for comparison, add "--greedy" in args.

# inference
numactl -m <node N> -C <cpu list> \
    python run_gptj.py \
        --precision <fp32/bf16/int8> \
        --max-new-tokens 32 \
        --num-iter 10 \
        --num-warmup 3 \
        --generation \
        --performance
```

### Language Modeling
>**Note**: Please apply gen-patch.patch to transforms as follow if you want to benchmark `int8` performance.
```bash
# only need to do when precision is int8.
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 9d1116e9951686f937d17697820117636bfc05a5
git apply clm-patch.patch
python setup.py develop
```
#### Run
```bash
# use jemalloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# for casual language modeling accuracy_only benchmark
numactl -m <node N> -C <cpu list> \
    python run_gptj.py \
        --precision <fp32/bf16/int8> \
        --clm \
        --accuracy_only

# for casual language modeling performance benchmark
numactl -m <node N> -C <cpu list> \
    python run_gptj.py \
        --precision <fp32/bf16> \
        --num_iter 10 \
        --num-warmup 3 \
        --batch-size 1 \
        --clm \
        --performance

```

## BLOOM-176B

We don't enable jemalloc here since BLOOM-176B requires lots of memory and will have memory contention w/ jemalloc.

```bash
numactl -m <node N> -C <cpu list> python3 run_bloom.py --batch_size 1 --benchmark
```
By default searcher is set to beam searcher with num_beams = 4, if you'd like to use greedy search for comparison, add "--greedy" in args.



  >**Note**: Inference performance speedup with Intel DL Boost (VNNI/AMX) on Intel(R) Xeon(R) hardware, Please refer to [Performance Tuning Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html) for more optimizations.
