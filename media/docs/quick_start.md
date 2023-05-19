# Quick Start

- [Notes](#notes)
- [Install Dependencies](#install-dependencies)
- [Setup Build Environment](#setup-build-environment)
- [Build Tests And Examples](#build-tests-and-examples)
- [Run Tests](#run-tests)
- [Run Examples](#run-examples)

## Notes
**${REPO_ROOT}** is the root directory of Intel® XeTLA git repo.

## Install Dependencies
```bash
$ sudo apt install libgtest-dev cmake
```

## Setup Build Environment
- Set up the environment for Intel oneAPI DPC++/C++ Compiler using the env.sh script.
- The command below assumes you installed to the default folder. If you customized the installation folder, please update ONEAPI_INSTALL_PATH in [env.sh](../../tools/scripts/env.sh) to your custom folder.
```bash
$ cd ${REPO_ROOT}
$ source tools/scripts/env.sh
```
## Build Tests And Examples
```bash
$ cd ${REPO_ROOT}
$ mkdir build && cd build
$ cmake ..
$ make -j
```

## Run Tests
```bash
$ cd ${REPO_ROOT}/build
$ ctest
```

you will see test output like:

```
 1/30 Test  #1: vector_add_tf32_1d ...............   Passed    0.23 sec
      Start  2: vector_add_bf16_2d
 2/30 Test  #2: vector_add_bf16_2d ...............   Passed    0.11 sec

 ......

29/30 Test #29: reg_reduce .......................   Passed    0.11 sec
      Start 30: math_general
30/30 Test #30: math_general .....................   Passed    0.15 sec

100% tests passed, 0 tests failed out of 30

Label Time Summary:
integration    =   5.92 sec*proc (11 tests)
unit           =   2.72 sec*proc (19 tests)

Total Test time (real) =   8.65 sec
```

## Run Examples
```bash
$ cd ${REPO_ROOT}/build
$ examples/01_basic_gemm/basic_gemm
```
you will see output like:

```
Running on Intel(R) Graphics [0x0bd5]
Bfloat16 GEMM Perforamnce:
        max relative diff:
                data_idx: 5 gold_idx: 5 relerr: 0.00775194
                data_val: 1032 gold_val: 1024
        max absolute diff:
                data_idx: 5 gold_idx: 5 abserr: 8
                data_val: 1032 gold_val: 1024
        max absolute ULP diff:
                data_idx: 1 gold_idx: 1 abserr: 1
                data_val: 17534 gold_val: 17533
        pass rate: 100%
PASSED

 Performance GPU time = 6.424176 ms, Gflops = 21394.019962
```
**Tips**  
The example demonstrates programming kernel with XeTLA, it works as expected with current configurations. Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors. Please contact us for support.