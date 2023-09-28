[README](/README.md#documentation) > **Quick Start**
# Quick Start

- [Preparations](/media/docs/quick_start.md#preparations)
- [Setup Environment](/media/docs/quick_start.md#setup-environment)
- [Build](/media/docs/quick_start.md#build)
- [Run Tests](/media/docs/quick_start.md#run-tests)
- [Run Examples](/media/docs/quick_start.md#run-examples)

## Preparations

- [Install Ubuntu](https://ubuntu.com/tutorials) version [22.04](http://releases.ubuntu.com/22.04/) and `packages`.

  ```bash
  $ sudo apt install libgtest-dev cmake
  ```

- [Install Intel GPU driver](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series) version [Stable 602](https://dgpu-docs.intel.com/releases/stable_602_20230323.html) or later.

- [Install Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) version [2023.2](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) or later.

  Select all components and default installation location when install, the default installation location is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts.

## Setup Environment
- Setup the environment for Intel oneAPI DPC++ Compiler.
- `ONEAPI_INSTALL_PATH` is `/opt/intel/oneapi` by default, if you customized the installation location, please update `ONEAPI_INSTALL_PATH` in [env.sh](/tools/scripts/env.sh).
- `${XETLA_REPO}` is the root directory of Intel® XeTLA git repo.

```bash
$ cd ${XETLA_REPO}
$ source tools/scripts/env.sh
```

## Build
```bash
$ cd ${XETLA_REPO}
$ mkdir build && cd build
$ cmake ..    # release version default, and 'cmake .. -DDEBUG=on' for debug version
$ make -j
```

## Run Tests
```bash
$ cd ${XETLA_REPO}/build
$ ctest
```

output like:

```
 1/30 Test  #1: vector_add_tf32_1d ...............   Passed    0.23 sec
      Start  2: vector_add_bf16_2d
 2/30 Test  #2: vector_add_bf16_2d ...............   Passed    0.11 sec

 ......

33/34 Test #33: math_general .....................   Passed    0.13 sec
      Start 34: epilogue_tile_op
34/34 Test #34: epilogue_tile_op .................   Passed    0.23 sec

100% tests passed, 0 tests failed out of 34

Label Time Summary:
integration    =  13.92 sec*proc (14 tests)
unit           =   2.62 sec*proc (20 tests)

Total Test time (real) =  16.55 sec
```

## Run Examples
```bash
$ cd ${XETLA_REPO}/build
$ examples/01_gemm_universal/gemm_universal
```
output like:

```
Running on Intel(R) Data Center GPU Max 1550
Local range: {1, 8, 4}
Group range: {1, 16, 16}
gemm validation:
        max relative diff:
                data_idx: 22 gold_idx: 22 relerr: 0.00775194
                data_val: 1032 gold_val: 1024
        max absolute diff:
                data_idx: 12 gold_idx: 12 abserr: 8
                data_val: 1048 gold_val: 1040
        max absolute ULP diff:
                data_idx: 1 gold_idx: 1 abserr: 1
                data_val: 17534 gold_val: 17533
        pass rate: 100%
PASSED

***************** PROFILING FOR KERNEL0 ***********************
============= Profiling for [kernel time] =============
[kernel time]The first   running(GPU_time) time is 6.63184ms
[kernel time]The minimum running(GPU_time) time is 6.60304ms
[kernel time]The maximum running(GPU_time) time is 7.01024ms
[kernel time]The median  running(GPU_time) time is 6.69744ms
[kernel time]The mean(exclude the first trial) running(GPU_time) time is 6.74343ms
[kernel time]The variance(exclude the first trial) running(GPU_time) time is 0.0157403ms
======================================================
============== [kernel time] gflops   ==================
[kernel time]The minimum gflops(GPU_time) is 19605.5
[kernel time]The maximum gflops(GPU_time) is 20814.5
[kernel time]The median  gflops(GPU_time) is 20521.1
[kernel time]The mean    gflops(GPU_time) is 20381.2
======================================================
```
`Notes`: The [examples](/examples) demonstrate programming kernels with Intel® XeTLA, it works as expected with current configurations. Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors. Please contact us for support.

## Copyright
Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
