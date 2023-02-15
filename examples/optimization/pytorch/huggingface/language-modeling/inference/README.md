Step-by-Step
============
This document describes the step-by-step instructions to run LLMs on 4th Gen Intel® Xeon® Scalable Processor (SPR) with PyTorch and Intel® Extension for PyTorch.

We now support 2 models, and we are adding more models and more advanced techniques(distributed inference, model compressions etc.) to better unleash LLM inference on Intel platforms.

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

```bash
# use jemalloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# default is beam search with num_beams=4, if you need to use greedy search for comparison, add "--greedy" in args.
numactl -m <node N> -C <cpu list> \
    python run_gptj.py \
        --precision <fp32/bf16> \
        --max-new-tokens 32
```
## BLOOM-176B
We don't enable jemalloc here since BLOOM-176B requires lots of memory and will have memory contention w/ jemalloc.

```bash
numactl -m <node N> -C <cpu list> python3 run_bloom.py --batch_size 1 --benchmark
```
By default searcher is set to beam searcher with num_beams = 4, if you'd like to use greedy search for comparison, add "--greedy" in args.



>**Note:** Inference performance speedup with Intel DL Boost (VNNI/AMX) on Intel(R) Xeon(R) hardware, Please refer to [Performance Tuning Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html) for more optimizations.
