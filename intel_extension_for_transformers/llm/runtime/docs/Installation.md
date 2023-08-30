Installation
=====
Support Windows and Linux operating systems for now.


## Prerequisites

- Python version: 3.6 or above
- GCC compiler: 7.2.1 or above
- CMake: 3.12 or above

## Install

Neural Engine is the backend of intel_extension_for_transformers. Installing intel_extension_for_transformers will build the binary and engine interface. More [details](https://github.com/intel/intel-extension-for-transformers/blob/main/README.md) are listed here.

### Install stable version intel_extension_for_transformers from pip
```shell
pip install intel-extension-for-transformers
```
### Install Neural Engine binary to deploy bare metal engine
```shell
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
git submodule update --init --recursive
cd intel_extension_for_transformers/backends/neural_engine/
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3) -DNE_WITH_SPARSELIB=True
make -j
```
After the steps above, `neural_engine_bin`, `neural_engine_py.cpython-37m-x86_64-linux-gnu.so`, `libkernellibs.so` and `libneural_engine.so` will be found in the `lib` subdirectory of the build folder. The first one is used for pure C++ model inference, and the second is used for python inference, they all depend on `libneural_engine.so` and `libkernellibs.so`.
