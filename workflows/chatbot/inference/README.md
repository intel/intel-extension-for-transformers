Chat with the NeuralChat
============

This document showcases the utilization of Llama2 model for conversing with NeuralChat. The inference of the models has been validated on the 4th Gen Intel® Xeon® Processors. We provide the [generate.py](./generate.py) script for performing inference on Intel® CPUs. We have enabled IPEX INT8 to speed up the inference. Please use the following commands for inference.


# Llama2

## Quantization and Inference on Xeon SPR

For [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) inference, we need use branch [int8_llama2](https://github.com/intel/intel-extension-for-transformers/tree/int8_llama2/workflows/chatbot/inference) and install IPEX `llm_feature_branch` branch, Please follow these steps to get the quantized model and do inference.
### Setup
```bash
WORK_DIR=$PWD
# GCC 12.3 is required, please set it firstly
# Create environment (conda recommended)
conda create -n llm python=3.9 -y
# install deps
conda install gcc=12.3 gxx=12.3 cxx-compiler -c conda-forge -y
conda install cmake ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

# Install PyTorch
python -m pip install torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu

# Install IPEX with semi-compiler, require gcc 12.3
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

# Install transformers
pip install transformers==4.31.0
# Install others deps
pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3
# Install intel-extension-for-transformers
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers
git checkout int8_llama2
python setup.py install

````
### Quantization
`meta-llama/Llama-2-7b-chat-hf` model need request the access, please follow the [instruction](https://huggingface.co/meta-llama/Llama-2-7b-hf), the quantized model saved in the `saved_results` folder and named `best_model.pt`.

```bash
cd intel-extension-for-transformers/workflows/chatbot/inference
python run_llama_int8.py \
        -m meta-llama/Llama-2-7b-chat-hf \
        --ipex-smooth-quant \
        --dataset "NeelNanda/pile-10k" \
        --output-dir "saved_results" \
        --jit \
        --int8-bf16-mixed
```

### Inference

```bash
cd intel-extension-for-transformers/workflows/chatbot/inference
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

# float32 and bfloat16
# dtype support "float32" and "bfloat16", default is "bfloat16"
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python  generate.py \
        --base_model_path meta-llama/Llama-2-7b-chat-hf \
        --use_kv_cache \
        --instructions "Tell me about Intel Xeon." \
        --dtype "float32"

# int8
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python  generate.py \
        --base_model_path meta-llama/Llama-2-7b-chat-hf \
        --use_kv_cache \
        --instructions "Tell me about Intel Xeon." \
        --ipex_int8 \
        --quantized_model_path "./saved_results/best_model.pt"
```




