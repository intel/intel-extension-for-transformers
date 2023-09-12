# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B),  [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf), [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1), [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b), [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b), [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b), [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat), [Intel/neural-chat-7b-v1-1](https://huggingface.co/Intel/neural-chat-7b-v1-1), more models are working in progress.

>**Note**: The default search algorithm is beam search with num_beams = 4, if you'd like to use greedy search for comparison, add "--greedy" in args.


# Prerequisite​
## 1. Create Environment​
If you want to use Pytorch & Intel-extension-for-pytorch version 2.0.1, please 
```
pip install -r requirements.txt
```
If you want to use Pytorch & Intel-extension-for-pytorch version 2.1, the dependent packages are listed in requirements, we recommend create environment as the following steps.

```bash
WORK_DIR=$PWD
# GCC 12.3 is required, please set it firstly
# Create environment (conda recommended)
conda create -n llm python=3.9 -y
# install deps, please try gcc, gxx 12.2 if 12.3 doesn't find from conda
conda install gcc=12.3 gxx=12.3 cxx-compiler -c conda-forge -y
conda install cmake ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

# Install PyTorch
python -m pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.1.0.dev20230711%2Bcpu-cp39-cp39-linux_x86_64.whl

# Install IPEX with semi-compiler, require gcc 12.3 or 12.2
rm -rf llvm-project && mkdir llvm-project && cd llvm-project
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/cmake-16.0.6.src.tar.xz
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-16.0.6.src.tar.xz
tar -xf cmake-16.0.6.src.tar.xz && mv cmake-16.0.6.src cmake
tar -xf llvm-16.0.6.src.tar.xz && mv llvm-16.0.6.src llvm
mkdir build && cd build
cmake ../llvm -DCMAKE_INSTALL_PREFIX=${PWD}/_install/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
make install -j$(nproc)
ln -s ${PWD}/_install/llvm/bin/llvm-config ${CONDA_PREFIX}/bin/llvm-config-13
cd ../../

git clone --branch llm_feature_branch https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git submodule sync && git submodule update --init --recursive
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py install
cd ../

# disable semi-compiler to avoid accuracy regression for mpt and neural-chat-v1-1 models, other models don't need it.
export _DNNL_DISABLE_COMPILER_BACKEND=1

# Install neural-compressor
git clone https://github.com/intel/neural-compressor.git
cd  neural-compressor
pip install -r requirements.txt
python setup.py install

# Install lm_eval
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@83dbfbf6070324f3e5872f63e49d49ff7ef4c9b3
# Install others deps
pip install transformers optimum-intel cpuid accelerate datasets sentencepiece protobuf==3.20.3
````
We use the GPTJ defination script [modeling_gptj.py](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/transformers/modeling/gptj/modeling_gptj.py) in `run_generation.py`. Here is a little change to success trace.
```diff
# Line 602 in modeling_gptj.py on transformers 4.28.1

-   position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
+   position_ids = torch.arange(past_length, torch.tensor(input_shape[-1]) + torch.tensor(past_length), dtype=torch.long, device=device)
```
The changes for `llama` series models in `modeling_llama.py`, `dolly_v2_3b` series models in `modeling_gpt_neox.py`， `bloom` series models in `modeling_bloom.py` and `opt` series models in `modeling_opt.py` are similar to the above.


# Run

## 1. Quantization
``` bash
# --int8 is used for int8 only.
# --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --quantize \
    --sq \
    --alpha 1.0 \
    --int8_bf16_mixed \
    --ipex
```
## 2. Performance
```bash
# --int8 is used for int8 only.
# --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --benchmark \
    --int8_bf16_mixed \
    --ipex
```
## 3. Accuracy
```bash
# --int8 is used for int8 only.
# --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
python run_generation.py \
   --model EleutherAI/gpt-j-6b \
   --accuracy \
   --int8_bf16_mixed \
   --ipex \
   --tasks "lambada_openai"
```
