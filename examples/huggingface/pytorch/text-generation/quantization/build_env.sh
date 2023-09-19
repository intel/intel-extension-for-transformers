#!/bin/bash
  2 set -x
  3 set -e
  4
  5 VER_LLVM="llvmorg-16.0.6"
  6 VER_IPEX="7256d0848ba81bb802dd33fca0e33049a751db58"
  7
  8 # Check existance of required Linux commands
  9 for CMD in conda git nproc make; do
 10     command -v ${CMD} || (echo "Error: Command \"${CMD}\" not found." ; exit 4)
 11 done
 12
 13 MAX_JOBS_VAR=$(nproc)
 14 if [ ! -z "${MAX_JOBS}" ]; then
 15     MAX_JOBS_VAR=${MAX_JOBS}
 16 fi
 17
 18 # Save current directory path
 19 BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
 20 cd ${BASEFOLDER}
 21 # Checkout individual components
 22 if [ ! -d llvm-project ]; then
 23     git clone https://github.com/llvm/llvm-project.git
 24 fi
 25 if [ ! -d intel-extension-for-pytorch ]; then
 26     git clone https://github.com/intel/intel-extension-for-pytorch.git
 27 fi
 28
 29 # Checkout required branch/commit and update submodules
 30 cd llvm-project
 31 if [ ! -z ${VER_LLVM} ]; then
 32     git checkout ${VER_LLVM}
 33 fi
 34 git submodule sync
 35 git submodule update --init --recursive
 36 cd ..
 37 cd intel-extension-for-pytorch
 38 if [ ! -z ${VER_IPEX} ]; then
 39     git checkout ${VER_IPEX}
 40 fi
 41 git submodule sync
 42 git submodule update --init --recursive
 43 cd ..
 44
 45 # Install dependencies
 46 conda install -y gcc==12.3 gxx==12.3 cxx-compiler -c conda-forge
 47 conda update -y sysroot_linux-64
 48 python -m pip install cmake
 49 python -m pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.1.0.dev20230711%2Bcpu-cp39-cp39-linux_x86_64.whl
 50 ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
 51
 52 # Compile individual component
 53 export CC=${CONDA_PREFIX}/bin/gcc
 54 export CXX=${CONDA_PREFIX}/bin/g++
 55 export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

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

