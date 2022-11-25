## Installation

Support Windows and Linux operating system for now.


### 0. Prerequisites

```
Python version: greater than 3.6

GCC compiler: 7.2.1 or above

CMake: 3.12 or above
```

### 1. Install neural-compressor

Neural Engine is the backend of intel_extension_for_transformers, just install intel_extension_for_transformers will build the binary and engine interface, more [detail](https://github.com/intel/intel-extension-for-transformers/blob/main/README.md).

### Install stable version from pip
```
pip install intel-extension-for-transformers
```
### Only install Neural Engine binary by deploy bare metal engine
```
git clone https://github.com/intel/intel-extension-for-transformers.git
cd <repo_path>
git submodule update --init --recursive
cd intel_extension_for_transformers/backends/neural_engine/
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3) -DNE_WITH_SPARSELIB=True
make -j
```
Then in the build folder, you will get the `neural_engine_bin`, `neural_engine_py.cpython-37m-x86_64-linux-gnu.so`, `libkernellibs.so` and `libneural_engine.so`. The first one is used for pure c++ model inference, and the second is used for python inference, they all need the `libneural_engine.so` and `libkernellibs.so`.
