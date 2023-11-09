# Text Generation
We provide the inference benchmarking script `run_generation.py` for large language models text generation.
Support most of large language models, such as GPT-J, LLaMa, OPT, BLOOM and etc.

## Setup
```bash
# Create Environment (conda recommended)
conda create -n llm python=3.9 -y
conda install ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

# install PyTorch
git clone https://github.com/pytorch/pytorch
cd pytorch
pip install -r requirements.txt
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
cd ../
# install
git clone https://github.com/intel/intel-extension-for-pytorch
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive
python setup.py install
cd ../
# install others deps
pip install cpuid transformers accelerate onnx sentencepiece
pip install --no-deps optimum
pip install --no-deps git+https://github.com/huggingface/optimum-intel.git@main
```

## Performance
```bash
# Setup Environment Variables for best performance on Xeon
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

# support single socket and multiple sockets
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_generation.py -m EleutherAI/gpt-j-6b --dtype bfloat16 --ipex
```

## Performance for GPU with DeepSpeed

Note: need ipex and torch-ccl with xpu support.

We recommend using `mpirun` to launch multi ranks inference rather than `deepspeed` launcher for better performance.

```bash
# run inference for bloom-176b
mpirun -np 8 --prepend-rank python -u run_generation_with_deepspeed.py -m bigscience/bloom --benchmark --ipex --input-tokens=1024 --max-new-tokens=128 --greedy

# run inference for bloom-176b with deepspeed launcher
deepspeed --num_gpus 8 --master_addr `hostname -I | sed -e 's/\s.*$//'` run_generation_with_deepspeed.py -m bigscience/bloom --benchmark --ipex --input-tokens=1024 --max-new-tokens=128 --greedy

# run inference for multi configs and only load checkpoint once
#mpirun -np 8 --prepend-rank python -u run_generation_with_deepspeed.py -m bigscience/bloom --benchmark --ipex --input-tokens 1024 1024 32 32 --max-new-tokens 128 128 32 32 --num-beams 1 4 1 4

# run accuracy test
LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m EleutherAI/gpt-j-6b --accuracy-only --ipex
```
