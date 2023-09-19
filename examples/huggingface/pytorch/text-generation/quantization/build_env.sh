#!/bin/bash
set -x
set -e

VER_LLVM="llvmorg-16.0.6"
VER_IPEX="7256d0848ba81bb802dd33fca0e33049a751db58"

# Check existance of required Linux commands
for CMD in conda git nproc make; do
    command -v ${CMD} || (echo "Error: Command \"${CMD}\" not found." ; exit 4)
done

MAX_JOBS_VAR=$(nproc)
if [ ! -z "${MAX_JOBS}" ]; then
    MAX_JOBS_VAR=${MAX_JOBS}
fi

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}
# Checkout individual components
if [ ! -d llvm-project ]; then
    git clone https://github.com/llvm/llvm-project.git
fi
if [ ! -d intel-extension-for-pytorch ]; then
    git clone https://github.com/intel/intel-extension-for-pytorch.git
fi

# Checkout required branch/commit and update submodules
cd llvm-project
if [ ! -z ${VER_LLVM} ]; then
    git checkout ${VER_LLVM}
fi
git submodule sync
git submodule update --init --recursive
cd ..
cd intel-extension-for-pytorch
if [ ! -z ${VER_IPEX} ]; then
    git checkout ${VER_IPEX}
fi
git submodule sync
git submodule update --init --recursive
cd ..

# Install dependencies
conda install -y gcc==12.3 gxx==12.3 cxx-compiler -c conda-forge
conda update -y sysroot_linux-64
python -m pip install cmake
python -m pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.1.0.dev20230711%2Bcpu-cp39-cp39-linux_x86_64.whl
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# Compile individual component
export CC=${CONDA_PREFIX}/bin/gcc
export CXX=${CONDA_PREFIX}/bin/g++
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

#  LLVM
cd llvm-project
LLVM_ROOT="$(pwd)/release"
if [ -d ${LLVM_ROOT} ]; then
    rm -rf ${LLVM_ROOT}
fi
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build
echo "***************************** cmake *****************************" > ../build.log
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF ../llvm 2>&1 | tee -a ../build.log
echo "***************************** build *****************************" >> ../build.log
cmake --build . -j ${MAX_JOBS_VAR} 2>&1 | tee -a ../build.log
echo "**************************** install ****************************" >> ../build.log
cmake -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT} -P cmake_install.cmake 2>&1 | tee -a ../build.log
#xargs rm -rf < install_manifest.txt
cd ..
rm -rf build
ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
cd ..
#  Intel® Extension for PyTorch*
cd intel-extension-for-pytorch
python -m pip install -r requirements.txt
export LLVM_DIR=${LLVM_ROOT}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
CXXFLAGS_BK=${CXXFLAGS}
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
export CXXFLAGS=${CXXFLAGS_BK}
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
python -m pip install --force-reinstall dist/*.whl
cd ..

# Sanity Test
set +x
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so
echo "Note: Should you experience \"version \`GLIBCXX_N.N.NN' not found\" error, run command \"export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so\" and try again."
python -c "import torch; import intel_extension_for_pytorch as ipex; print(f'torch_cxx11_abi:     {torch._C._GLIBCXX_USE_CXX11_ABI}'); print(f'torch_version:       {torch.__version__}'); print(f'ipex_version:        {ipex.__version__}');"
# Install neural-compressor
git clone https://github.com/intel/neural-compressor.git
cd  neural-compressor
pip install -r requirements.txt
python setup.py install
cd ..

# Install intel-extension-for-pytorch
git checkout -b int8_llama2
pip install -r requirements.txt
python setup.py install
cd ..

# Install lm_eval
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@83dbfbf6070324f3e5872f63e49d49ff7ef4c9b3
# Install others deps
pip install transformers optimum-intel cpuid accelerate datasets sentencepiece protobuf==3.20.3
